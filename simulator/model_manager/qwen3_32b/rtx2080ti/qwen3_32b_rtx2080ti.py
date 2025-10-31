# 2024.10.27 Qwen3-32B on RTX2080Ti

import os
from typing import Dict

from simulator.model_manager.base_classes import ModelOnMachine
from simulator.model_manager.qwen3_32b.helper import qwen32b_workload_ratio, qwen32b_typical_statistics
from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import VLLM_BLOCK_SIZE, MAX_INPUT_LEN, DECODE_PER_TOKEN_MAX_CONTEXT


class Qwen32BonRTX2080Ti(ModelOnMachine):
    def __init__(self, num_machines_dict: Dict[str, int], typical_layers_dict: Dict[str, int],
                 normalized_perf_dict: Dict[str, float]):
        """
        Qwen3-32B + RTX2080Ti 11GB
        Qwen3-32B has 64 layers, approximately 32B parameters
        Each layer is about 0.5GB (32B / 64 layers)
        RTX2080Ti has 11GB VRAM, can fit approximately 10-11 layers with KV cache
        """
        # --------------- Machine Dependent Data --------------- #
        machine_name: str = "RTX2080Ti"
        max_num_layers: int = 10  # Can fit 10 layers of Qwen3-32B per 11GB GPU (for 7 GPUs setup)
        
        # Qwen3-32B has 64 layers, larger than Qwen2.5-14B (48 layers)
        # With 7 GPUs and 9-10 layers each, we can cover all 64 layers
        # First and last GPUs handle embedding/output, so they can have fewer layers
        # Adjusted KV cache blocks based on memory constraints
        vllm_num_blocks_dict: Dict[int, int] = {
            1: 30000, 2: 15000, 3: 10000, 4: 7500, 5: 6000,
            6: 5000, 7: 4200, 8: 3600, 9: 3200, 10: 2800
        }
        prompt_max_requests_dict: Dict[int, int] = {
            1: 2, 2: 2, 3: 1, 4: 1, 5: 1,
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1
        }
        decode_max_tokens_dict: Dict[int, int] = {
            1: 400, 2: 280, 3: 200, 4: 150, 5: 120,
            6: 100, 7: 85, 8: 70, 9: 60, 10: 50
        }
        # ------------------------------------------------------ #

        # save num machines dict
        self.machine_name: str = machine_name
        self.num_machines_dict: Dict[str, int] = num_machines_dict

        # Profiling data for Qwen3-32B on RTX2080Ti
        # Based on Qwen2.5-14B but scaled for larger model (32B vs 14B)
        # Inference time increases approximately 2.2x due to larger intermediate_size (27648 vs 13824)
        # and more layers (64 vs 48)
        self.prompt_bs2time: Dict[int, float] = {
            128: 0.30, 256: 0.52, 384: 0.74, 512: 0.96,
            640: 1.18, 768: 1.40, 896: 1.62, 1024: 1.84,
            1152: 2.06, 1280: 2.28, 1408: 2.50, 1536: 2.72,
            1664: 2.94, 1792: 3.16, 1920: 3.38, 2048: 3.60
        }
        self.prompt_bs2vram: Dict[int, float] = {
            bs: 0 for bs in self.prompt_bs2time
        }
        self.decode_bs2time: Dict[int, float] = {
            1: 0.030, 2: 0.037, 3: 0.044, 4: 0.050, 5: 0.057,
            10: 0.083, 20: 0.135, 30: 0.188, 40: 0.240, 50: 0.293,
            60: 0.345, 70: 0.398, 80: 0.450, 90: 0.503, 100: 0.555,
            150: 0.818, 200: 1.080, 250: 1.343, 300: 1.605, 350: 1.868
        }
        self.decode_bs2vram: Dict[int, float] = {
            bs: 0 for bs in self.decode_bs2time
        }

        # kv cache & activation backup cache
        self.max_num_layers: int = max_num_layers
        self.kv_cache_capacity: Dict[int, int] = {
            _num_layers: VLLM_BLOCK_SIZE * _num_blocks * _num_layers for _num_layers, _num_blocks in
            vllm_num_blocks_dict.items()
        }
        self.activation_backup_capacity: Dict[int, int] = {
            _num_layers: 0 for _num_layers in self.kv_cache_capacity
        }

        # build the inference settings
        self.num_layers_to_inference_settings: Dict[int, InferenceSettings] = {}
        for cur_num_layers in range(1, self.max_num_layers + 1):
            cur_workload_ratio = qwen32b_workload_ratio(
                target_machine_name=machine_name,
                target_num_layers=cur_num_layers,
                num_machines_dict=num_machines_dict,
                typical_layers_dict=typical_layers_dict,
                normalized_perf_dict=normalized_perf_dict
            )
            prompt_typical_requests, prompt_typical_tokens, decode_typical_tokens = qwen32b_typical_statistics(
                workload_ratio=cur_workload_ratio,
                num_kv_cache_entries=self.kv_cache_capacity[cur_num_layers],
                num_layers_on_node=cur_num_layers
            )
            self.num_layers_to_inference_settings[cur_num_layers] = InferenceSettings(
                prompt_max_requests=prompt_max_requests_dict[cur_num_layers],
                prompt_max_tokens=prompt_max_requests_dict[cur_num_layers] * MAX_INPUT_LEN,
                prompt_typical_requests=min(prompt_typical_requests, 1.0),
                prompt_typical_tokens=prompt_typical_tokens,
                decode_max_context=decode_max_tokens_dict[cur_num_layers] * DECODE_PER_TOKEN_MAX_CONTEXT,
                decode_max_tokens=decode_max_tokens_dict[cur_num_layers],
                decode_typical_tokens=decode_typical_tokens
            )

    def get_profiling_results(self) -> MachineProfile:
        """Get the profiling results of running one layer of the model on the machine."""
        machine_profile = MachineProfile(
            prompt_bs2time=self.prompt_bs2time,
            prompt_bs2vram=self.prompt_bs2vram,
            decode_bs2time=self.decode_bs2time,
            decode_bs2vram=self.decode_bs2vram
        )
        return machine_profile

    def get_max_num_layers(self) -> int:
        """Get the max number of layers that can be loaded into this machine."""
        return self.max_num_layers

    def get_inference_settings(self, num_on_node_layers: int) -> InferenceSettings:
        """Get the inference settings when there are given number of layers on node."""
        assert 0 < num_on_node_layers <= self.max_num_layers, "Bad number of layers on node!"
        return self.num_layers_to_inference_settings[num_on_node_layers]

    def get_typical_token_throughput(self, num_on_node_layers: int) -> float:
        """Get typical token throughput when there are given number of layers on node."""
        inference_settings = self.get_inference_settings(num_on_node_layers=num_on_node_layers)
        prompt_typical_requests = inference_settings.prompt_typical_requests
        prompt_typical_tokens = inference_settings.prompt_typical_tokens
        decode_typical_tokens = inference_settings.decode_typical_tokens

        # Calculation based on qwen2_5_14b implementation
        from simulator.event_simulator.utils import linear_interpolate

        def _get_prompt_time(prompt_num_tokens: int) -> float:
            if prompt_num_tokens <= 0:
                return 0.0
            prompt_left, prompt_right = -1, 1000 * 1000
            for prompt_point in self.prompt_bs2time:
                if prompt_left < prompt_point <= prompt_num_tokens:
                    prompt_left = prompt_point
                if prompt_num_tokens <= prompt_point < prompt_right:
                    prompt_right = prompt_point
            # Handle edge cases
            if prompt_left == -1:
                prompt_left = min(self.prompt_bs2time.keys())
            if prompt_right == 1000 * 1000:
                prompt_right = max(self.prompt_bs2time.keys())
            # Handle exact matches
            if prompt_num_tokens == prompt_left:
                return self.prompt_bs2time[prompt_left]
            if prompt_num_tokens == prompt_right:
                return self.prompt_bs2time[prompt_right]
            # Extrapolate if needed
            if prompt_num_tokens < prompt_left:
                return self.prompt_bs2time[prompt_left] * (prompt_num_tokens / prompt_left)
            if prompt_num_tokens > prompt_right:
                return self.prompt_bs2time[prompt_right] * (prompt_num_tokens / prompt_right)
            # Interpolate
            return linear_interpolate(
                x_0=prompt_left, y_0=self.prompt_bs2time[prompt_left],
                x_1=prompt_right, y_1=self.prompt_bs2time[prompt_right],
                x_target=prompt_num_tokens
            )

        def _get_decode_time(decode_num_tokens: int) -> float:
            if decode_num_tokens <= 0:
                return 0.0
            decode_left, decode_right = -1, 1000 * 1000
            for decode_point in self.decode_bs2time:
                if decode_left < decode_point <= decode_num_tokens:
                    decode_left = decode_point
                if decode_num_tokens <= decode_point < decode_right:
                    decode_right = decode_point
            # Handle edge cases
            if decode_left == -1:
                decode_left = min(self.decode_bs2time.keys())
            if decode_right == 1000 * 1000:
                decode_right = max(self.decode_bs2time.keys())
            # Handle exact matches
            if decode_num_tokens == decode_left:
                return self.decode_bs2time[decode_left]
            if decode_num_tokens == decode_right:
                return self.decode_bs2time[decode_right]
            # Extrapolate if needed
            if decode_num_tokens < decode_left:
                return self.decode_bs2time[decode_left] * (decode_num_tokens / decode_left)
            if decode_num_tokens > decode_right:
                return self.decode_bs2time[decode_right] * (decode_num_tokens / decode_right)
            # Interpolate
            return linear_interpolate(
                x_0=decode_left, y_0=self.decode_bs2time[decode_left],
                x_1=decode_right, y_1=self.decode_bs2time[decode_right],
                x_target=decode_num_tokens
            )

        # Calculation method is dependent on prompt typical requests
        if prompt_typical_requests >= 1:
            # in linear region, no need to rescale
            total_tokens = prompt_typical_tokens + decode_typical_tokens
            layer_prompt_time = _get_prompt_time(prompt_num_tokens=prompt_typical_tokens)
            layer_decode_time = _get_decode_time(decode_num_tokens=decode_typical_tokens)
            total_time = num_on_node_layers * (layer_prompt_time + layer_decode_time)
            if total_time > 0:
                return total_tokens / total_time
            else:
                return 100.0
        else:
            # need to scale to 1
            rescaling = 1 / prompt_typical_requests
            total_tokens = rescaling * (prompt_typical_tokens + decode_typical_tokens)
            layer_prompt_time = _get_prompt_time(prompt_num_tokens=int(prompt_typical_tokens * rescaling))
            layer_decode_time = _get_decode_time(decode_num_tokens=decode_typical_tokens) * rescaling
            total_time = num_on_node_layers * (layer_prompt_time + layer_decode_time)
            if total_time > 0:
                return total_tokens / total_time
            else:
                return 100.0

    def get_kv_cache_capacity(self, num_on_node_layers: int) -> int:
        """Get the kv cache capacity when there are given number of layers on node."""
        assert 0 < num_on_node_layers <= self.max_num_layers, "Bad number of layers on node!"
        return self.kv_cache_capacity[num_on_node_layers]

    def get_activation_backup_capacity(self, num_on_node_layers: int) -> int:
        """Get the activation backup capacity when there are given number of layers on node."""
        assert 0 < num_on_node_layers <= self.max_num_layers, "Bad number of layers on node!"
        return self.activation_backup_capacity[num_on_node_layers]
