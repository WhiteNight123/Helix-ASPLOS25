# 2024.11.06 Qwen3-14B on RTX4090

import os
from typing import Dict

from simulator.model_manager.base_classes import ModelOnMachine
from simulator.model_manager.qwen3_14b.helper import qwen3_14b_workload_ratio, qwen3_14b_typical_statistics
from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import VLLM_BLOCK_SIZE, MAX_INPUT_LEN, DECODE_PER_TOKEN_MAX_CONTEXT


class Qwen3_14BonRTX4090(ModelOnMachine):
    def __init__(self, num_machines_dict: Dict[str, int], typical_layers_dict: Dict[str, int],
                 normalized_perf_dict: Dict[str, float]):
        """
        Qwen3-14B + RTX4090 24GB
        Qwen3-14B has 40 layers, approximately 14B parameters
        Each layer is about 0.35GB (14B / 40 layers) in FP16
        RTX4090 has 24GB VRAM:
        - Without quantization: can fit approximately 16-20 layers with KV cache
        - RTX4090 is approximately 2x faster than RTX2080Ti in compute performance
        """
        # --------------- Machine Dependent Data --------------- #
        machine_name: str = "RTX4090"
        max_num_layers: int = 20  # Can fit 20 layers of Qwen3-14B per 24GB GPU
        
        # RTX4090 is approximately 2x faster than RTX2080Ti
        # Adjusted KV cache blocks based on larger memory capacity (24GB vs 11GB)
        vllm_num_blocks_dict: Dict[int, int] = {
            1: 60000, 2: 30000, 3: 20000, 4: 15000, 5: 12000,
            6: 10000, 7: 8500, 8: 7500, 9: 6700, 10: 6000,
            11: 5400, 12: 5000, 13: 4600, 14: 4300, 15: 4000,
            16: 3800, 17: 3600, 18: 3400, 19: 3200, 20: 3000
        }
        prompt_max_requests_dict: Dict[int, int] = {
            1: 3, 2: 3, 3: 2, 4: 2, 5: 2,
            6: 2, 7: 1, 8: 1, 9: 1, 10: 1,
            11: 1, 12: 1, 13: 1, 14: 1, 15: 1,
            16: 1, 17: 1, 18: 1, 19: 1, 20: 1
        }
        decode_max_tokens_dict: Dict[int, int] = {
            1: 800, 2: 600, 3: 400, 4: 300, 5: 240,
            6: 200, 7: 170, 8: 150, 9: 130, 10: 115,
            11: 105, 12: 95, 13: 88, 14: 82, 15: 77,
            16: 72, 17: 68, 18: 65, 19: 62, 20: 59
        }
        # ------------------------------------------------------ #

        # save num machines dict
        self.machine_name: str = machine_name
        self.num_machines_dict: Dict[str, int] = num_machines_dict

        # Profiling data for Qwen3-14B on RTX4090
        # RTX4090 is approximately 2x faster than RTX2080Ti
        # Adjusted times based on higher compute throughput and memory bandwidth
        self.prompt_bs2time: Dict[int, float] = {
            128: 0.07, 256: 0.12, 384: 0.17, 512: 0.22,
            640: 0.27, 768: 0.32, 896: 0.37, 1024: 0.42,
            1152: 0.47, 1280: 0.52, 1408: 0.57, 1536: 0.62,
            1664: 0.67, 1792: 0.72, 1920: 0.77, 2048: 0.82
        }
        self.prompt_bs2vram: Dict[int, float] = {
            bs: 0 for bs in self.prompt_bs2time
        }
        self.decode_bs2time: Dict[int, float] = {
            1: 0.007, 2: 0.008, 3: 0.010, 4: 0.011, 5: 0.013,
            10: 0.019, 20: 0.031, 30: 0.043, 40: 0.055, 50: 0.067,
            60: 0.079, 70: 0.091, 80: 0.103, 90: 0.115, 100: 0.127,
            150: 0.187, 200: 0.247, 250: 0.307, 300: 0.367, 400: 0.487
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
            cur_workload_ratio = qwen3_14b_workload_ratio(
                target_machine_name=machine_name,
                target_num_layers=cur_num_layers,
                num_machines_dict=num_machines_dict,
                typical_layers_dict=typical_layers_dict,
                normalized_perf_dict=normalized_perf_dict
            )
            prompt_typical_requests, prompt_typical_tokens, decode_typical_tokens = qwen3_14b_typical_statistics(
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

        # Calculation based on llama1_30b implementation
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
            return linear_interpolate(prompt_left, self.prompt_bs2time[prompt_left],
                                     prompt_right, self.prompt_bs2time[prompt_right],
                                     prompt_num_tokens)

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
            return linear_interpolate(decode_left, self.decode_bs2time[decode_left],
                                     decode_right, self.decode_bs2time[decode_right],
                                     decode_num_tokens)

        prompt_time = _get_prompt_time(prompt_typical_tokens)
        decode_time = _get_decode_time(decode_typical_tokens)
        total_time = prompt_time + decode_time

        total_tokens = prompt_typical_tokens + decode_typical_tokens

        if total_time > 0:
            return total_tokens / total_time
        return 0.0

    def get_kv_cache_capacity(self, num_on_node_layers: int) -> int:
        """Get the kv cache capacity of this machine when using the current model."""
        assert 0 < num_on_node_layers <= self.max_num_layers
        return self.kv_cache_capacity[num_on_node_layers]

    def get_activation_backup_capacity(self, num_on_node_layers: int) -> int:
        """Get the activation backup capacity of this machine when using the current model."""
        assert 0 < num_on_node_layers <= self.max_num_layers
        return self.activation_backup_capacity[num_on_node_layers]
