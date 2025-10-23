# 2024.11.02 LLaMa2-13B Statistics

from typing import List, Dict

from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.model_manager.base_classes import ModelStatistics
from simulator.model_manager.llama2_13b.rtx2080ti.llama2_13b_rtx2080ti import LLaMa13BonRTX2080Ti
from simulator.event_simulator.utils import Byte, GB, LLaMa2_13B_TOTAL_LAYERS


class LLaMa13BStatistics(ModelStatistics):
    def __init__(self, num_machines_dict: Dict[str, int]):
        """
        This class stores the profiling results of different machines running LLaMa2-13B.
        num_machines_dict is a subset of supported machine types.
        Currently supports: {"RTX2080Ti": X}
        """
        # estimate the typical number of layers on node
        typical_layers_dict = {"RTX2080Ti": 10}
        total_layer_capacity = 0
        for machine_name in num_machines_dict:
            total_layer_capacity += num_machines_dict[machine_name] * typical_layers_dict.get(machine_name, 10)
        
        # Adjust if capacity is low
        if total_layer_capacity < LLaMa2_13B_TOTAL_LAYERS * 1.2:
            typical_layers_dict = {"RTX2080Ti": 10}  # Maximum layers per GPU
        typical_layers_dict = {m_type: typical_layers_dict.get(m_type, 10) for m_type in num_machines_dict}

        # estimate the normalized performance
        normalized_perf_dict = {"RTX2080Ti": 15}  # Comparable to T4
        normalized_perf_dict = {m_type: normalized_perf_dict.get(m_type, 15) for m_type in num_machines_dict}
        
        # Iteratively refine performance estimates
        for iteration in range(10):
            new_normalized_perf_dict = {}
            if "RTX2080Ti" in num_machines_dict:
                rtx2080ti = LLaMa13BonRTX2080Ti(
                    num_machines_dict=num_machines_dict,
                    typical_layers_dict=typical_layers_dict,
                    normalized_perf_dict=normalized_perf_dict
                )
                rtx2080ti_typical_tp = rtx2080ti.get_typical_token_throughput(
                    num_on_node_layers=typical_layers_dict["RTX2080Ti"]
                )
                new_normalized_perf_dict["RTX2080Ti"] = rtx2080ti_typical_tp * typical_layers_dict["RTX2080Ti"]
            normalized_perf_dict = new_normalized_perf_dict

        # save the final results
        self.num_machines_dict: Dict[str, int] = num_machines_dict
        self.typical_layers_dict: Dict[str, int] = typical_layers_dict
        self.normalized_perf_dict: Dict[str, float] = normalized_perf_dict

        # machine profiling results
        if "RTX2080Ti" in num_machines_dict:
            self.rtx2080ti = LLaMa13BonRTX2080Ti(
                num_machines_dict=num_machines_dict,
                typical_layers_dict=typical_layers_dict,
                normalized_perf_dict=normalized_perf_dict
            )
        else:
            self.rtx2080ti = None

        # model statistics
        self.token_size: float = 2 * Byte  # FP16
        self.activation_size: float = 5120 * 2 * Byte  # hidden_size * 2 bytes
        # 13B model with 40 layers, approximately 0.65GB per layer
        self.model_param_sizes = [0.65 * GB] * 40
        assert len(self.model_param_sizes) == LLaMa2_13B_TOTAL_LAYERS, "Total layer number mismatch!"

    def check_type_exist(self, machine_type: str) -> bool:
        """Check if the given machine type exists in the current cluster."""
        return machine_type in self.num_machines_dict

    def get_profiling_results(self, machine_type: str) -> MachineProfile:
        """Get the profiling results of running one layer of the model on given type of machine."""
        assert self.check_type_exist(machine_type), f"Machine type {machine_type} not found!"
        if machine_type == "RTX2080Ti":
            return self.rtx2080ti.get_profiling_results()
        else:
            assert False, f"Unknown machine type: {machine_type}"

    def get_max_num_layers(self, machine_type: str) -> int:
        """Get the max number of layers the given type of machine can hold."""
        assert self.check_type_exist(machine_type), f"Machine type {machine_type} not found!"
        if machine_type == "RTX2080Ti":
            return self.rtx2080ti.get_max_num_layers()
        else:
            assert False, f"Unknown machine type: {machine_type}"

    def get_inference_settings(self, machine_type: str, num_on_node_layers: int) -> InferenceSettings:
        """Get the inference settings of the given machine type when there are given number of layers."""
        assert self.check_type_exist(machine_type), f"Machine type {machine_type} not found!"
        if machine_type == "RTX2080Ti":
            return self.rtx2080ti.get_inference_settings(num_on_node_layers=num_on_node_layers)
        else:
            assert False, f"Unknown machine type: {machine_type}"

    def get_typical_token_throughput(self, machine_type: str, num_on_node_layers: int) -> float:
        """Get the typical token throughput of given machine type when there are given number of layers on node."""
        assert self.check_type_exist(machine_type), f"Machine type {machine_type} not found!"
        if machine_type == "RTX2080Ti":
            return self.rtx2080ti.get_typical_token_throughput(num_on_node_layers=num_on_node_layers)
        else:
            assert False, f"Unknown machine type: {machine_type}"

    def get_kv_cache_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """Get the kv cache capacity of given machine type when using the current model."""
        assert self.check_type_exist(machine_type), f"Machine type {machine_type} not found!"
        if machine_type == "RTX2080Ti":
            return self.rtx2080ti.get_kv_cache_capacity(num_on_node_layers=num_on_node_layers)
        else:
            assert False, f"Unknown machine type: {machine_type}"

    def get_activation_backup_capacity(self, machine_type: str, num_on_node_layers: int) -> int:
        """Get the activation backup capacity of given machine type when using the current model."""
        assert self.check_type_exist(machine_type), f"Machine type {machine_type} not found!"
        if machine_type == "RTX2080Ti":
            return self.rtx2080ti.get_activation_backup_capacity(num_on_node_layers=num_on_node_layers)
        else:
            assert False, f"Unknown machine type: {machine_type}"

    # ----------------------------------------- Model ----------------------------------------- #

    def get_model_params(self) -> List[float]:
        """Get the param size list of the model."""
        return self.model_param_sizes

    def get_model_token_size(self) -> float:
        """Get the token size of the model."""
        return self.token_size

    def get_model_activation_size(self) -> float:
        """Get the activation size of the model."""
        return self.activation_size
