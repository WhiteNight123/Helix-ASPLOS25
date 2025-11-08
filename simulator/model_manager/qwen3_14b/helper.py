# 2024.11.06 Qwen3-14B Helper Functions

from typing import Dict
from simulator.event_simulator.utils import AVG_INPUT_LEN, AVG_OUTPUT_LEN


# 2024.11.06 Helper functions for Qwen3-14B

from typing import Dict, Tuple
from simulator.event_simulator.utils import Qwen3_14B_TOTAL_LAYERS, AVG_INPUT_LEN, AVG_OUTPUT_LEN, KV_CACHE_HWM


def qwen3_14b_workload_ratio(target_machine_name: str, target_num_layers: int,
                             num_machines_dict: Dict[str, int], typical_layers_dict: Dict[str, int],
                             normalized_perf_dict: Dict[str, float]) -> float:
    """
    Get the workload ratio of given machine with given number of layers.
    Note: workload_ratio = num_layers * t_machine / T, where T is the round-trip time of a request.
    """
    assert num_machines_dict.keys() == typical_layers_dict.keys() == normalized_perf_dict.keys(), "Keys mismatch!"

    # get round trip time
    layer_times_dict: Dict[str, float] = {_name: 1 / _perf for _name, _perf in normalized_perf_dict.items()}
    sum_of_time, sum_of_layers = 0, 0
    for cur_machine_name, cur_num_machine in num_machines_dict.items():
        cur_typical_layer = typical_layers_dict[cur_machine_name]
        cur_layer_time = layer_times_dict[cur_machine_name]
        sum_of_time += cur_num_machine * cur_typical_layer * cur_layer_time
        sum_of_layers += cur_num_machine * cur_typical_layer
    round_trip_time = (Qwen3_14B_TOTAL_LAYERS - target_num_layers) * (sum_of_time / sum_of_layers)

    # get batch time
    batch_time = target_num_layers * layer_times_dict[target_machine_name]
    round_trip_time += batch_time
    assert round_trip_time > batch_time, "Round trip time is smaller than batch time!"

    # get workload ratio
    workload_ratio = batch_time / round_trip_time
    return workload_ratio


def qwen3_14b_typical_statistics(workload_ratio: float, num_kv_cache_entries: int,
                                 num_layers_on_node: int) -> Tuple[float, int, int]:
    """
    Get prompt_typical_requests, prompt_typical_tokens, decode_typical_tokens
    """
    # get seq length & available kv cache entries
    seq_length: int = AVG_INPUT_LEN + AVG_OUTPUT_LEN
    available_layer_entries: int = int(KV_CACHE_HWM * num_kv_cache_entries / num_layers_on_node)

    # calculate typical batch size
    prompt_typical_requests: float = available_layer_entries / seq_length * workload_ratio / AVG_OUTPUT_LEN
    prompt_typical_tokens: int = round(prompt_typical_requests * AVG_INPUT_LEN)
    decode_typical_tokens: int = round(available_layer_entries / seq_length * workload_ratio)
    if prompt_typical_requests > 1:
        # already in linear region, just cap at 1
        prompt_typical_requests = 1
        prompt_typical_tokens = AVG_INPUT_LEN
        decode_typical_tokens = AVG_OUTPUT_LEN
    return prompt_typical_requests, prompt_typical_tokens, decode_typical_tokens


def qwen14b_typical_statistics(workload_ratio: float, num_kv_cache_entries: int,
                                num_layers_on_node: int) -> Tuple[float, int, int]:
    """
    Get prompt_typical_requests, prompt_typical_tokens, decode_typical_tokens
    """
    # get seq length & available kv cache entries
    seq_length: int = AVG_INPUT_LEN + AVG_OUTPUT_LEN
    available_layer_entries: int = int(KV_CACHE_HWM * num_kv_cache_entries / num_layers_on_node)

    # calculate typical batch size
    prompt_typical_requests: float = available_layer_entries / seq_length * workload_ratio / AVG_OUTPUT_LEN
    prompt_typical_tokens: int = round(prompt_typical_requests * AVG_INPUT_LEN)
    decode_typical_tokens: int = round(available_layer_entries / seq_length * workload_ratio)
    if prompt_typical_requests > 1:
        # already in linear region, just cap at 1
        prompt_typical_requests = 1
        prompt_typical_tokens = AVG_INPUT_LEN
        decode_typical_tokens = AVG_OUTPUT_LEN
    return prompt_typical_requests, prompt_typical_tokens, decode_typical_tokens
