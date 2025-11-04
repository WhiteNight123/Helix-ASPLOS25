# 2024.10.27 Qwen3-32B on RTX4090

import os
from typing import Dict

from simulator.model_manager.base_classes import ModelOnMachine
from simulator.model_manager.qwen3_32b.helper import qwen32b_workload_ratio, qwen32b_typical_statistics
from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import VLLM_BLOCK_SIZE, MAX_INPUT_LEN, DECODE_PER_TOKEN_MAX_CONTEXT


class Qwen32BonRTX4090(ModelOnMachine):
    def __init__(self, num_machines_dict: Dict[str, int], typical_layers_dict: Dict[str, int],
                 normalized_perf_dict: Dict[str, float]):
        """
        Qwen3-32B + RTX4090 24GB
        Qwen3-32B has 64 layers, approximately 32B parameters
        Each layer is about 0.5GB (32B / 64 layers) in FP16
        With AWQ quantization (4-bit), each layer is about 0.125GB
        RTX4090 has 24GB VRAM:
        - Without quantization: can fit approximately 24-26 layers with KV cache
        - With AWQ quantization: can fit approximately 32-40 layers with KV cache
        """
        # --------------- Machine Dependent Data --------------- #
        machine_name: str = "RTX4090"
        max_num_layers: int = 32  # Can fit 32 layers of Qwen3-32B-AWQ per 24GB GPU
        
        # With 24GB VRAM, RTX4090 can handle more layers than RTX2080Ti (11GB)
        # For 3 GPU setup: 2x RTX2080Ti (16 layers each) + 1x RTX4090 (32 layers)
        # RTX4090 is approximately 2x faster than RTX2080Ti in compute performance
        # Adjusted KV cache blocks based on larger memory capacity
        vllm_num_blocks_dict: Dict[int, int] = {
            1: 80000, 2: 40000, 3: 26000, 4: 20000, 5: 16000,
            6: 13000, 7: 11000, 8: 9500, 9: 8500, 10: 7500,
            11: 6800, 12: 6200, 13: 5700, 14: 5300, 15: 4900, 16: 4600,
            17: 4300, 18: 4100, 19: 3900, 20: 3700, 21: 3500, 22: 3300,
            23: 3200, 24: 3000, 25: 2900, 26: 2800, 27: 2700, 28: 2600,
            29: 2500, 30: 2400, 31: 2300, 32: 2200
        }
        prompt_max_requests_dict: Dict[int, int] = {
            1: 4, 2: 3, 3: 2, 4: 2, 5: 2,
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1,
            11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1,
            17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1,
            23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1,
            29: 1, 30: 1, 31: 1, 32: 1
        }
        decode_max_tokens_dict: Dict[int, int] = {
            1: 800, 2: 560, 3: 400, 4: 300, 5: 240,
            6: 200, 7: 170, 8: 150, 9: 130, 10: 115,
            11: 105, 12: 95, 13: 88, 14: 82, 15: 77, 16: 72,
            17: 68, 18: 65, 19: 62, 20: 59, 21: 56, 22: 54,
            23: 52, 24: 50, 25: 48, 26: 46, 27: 45, 28: 44,
            29: 43, 30: 42, 31: 41, 32: 40
        }
        # ------------------------------------------------------ #

        # save num machines dict
        self.machine_name: str = machine_name
        self.num_machines_dict: Dict[str, int] = num_machines_dict

        # Profiling data for Qwen3-32B on RTX4090
        # RTX4090 is approximately 2x faster than RTX2080Ti
        # Adjusted times based on higher compute throughput and memory bandwidth
        self.prompt_bs2time: Dict[int, float] = {
            128: 0.15, 256: 0.26, 384: 0.37, 512: 0.48,
            640: 0.59, 768: 0.70, 896: 0.81, 1024: 0.92,
            1152: 1.03, 1280: 1.14, 1408: 1.25, 1536: 1.36,
            1664: 1.47, 1792: 1.58, 1920: 1.69, 2048: 1.80
        }
        self.prompt_bs2vram: Dict[int, float] = {
            bs: 0 for bs in self.prompt_bs2time
        }
        self.decode_bs2time: Dict[int, float] = {
            1: 0.015, 2: 0.019, 3: 0.022, 4: 0.025, 5: 0.029,
            10: 0.042, 20: 0.068, 30: 0.094, 40: 0.120, 50: 0.147,
            60: 0.173, 70: 0.199, 80: 0.225, 90: 0.252, 100: 0.278,
            150: 0.409, 200: 0.540, 250: 0.672, 300: 0.803, 350: 0.934,
            400: 1.065, 450: 1.196, 500: 1.328
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
