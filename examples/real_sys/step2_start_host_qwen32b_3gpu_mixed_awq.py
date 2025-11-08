#!/usr/bin/env python3
# Start Host for Qwen3-32B-AWQ with 2x RTX2080Ti + 1x RTX4090
import os
import sys

from llm_sys.maxflow_host import run_maxflow_host_online, run_maxflow_host_offline
from llm_sys.heuristic_host import run_heuristic_host_online, run_heuristic_host_offline
from simulator.event_simulator.cluster_simulator import ModelName


def qwen32b_3gpu_mixed_awq_maxflow_offline():
    os.makedirs("./result/qwen32b_3gpu_mixed_awq_maxflow_offline/", exist_ok=True)
    print("Running Qwen3-32B-AWQ: maxflow host + offline mode (2x RTX2080Ti + 1x RTX4090)")
    run_maxflow_host_offline(
        # model and machine
        machine_num_dict={"RTX2080Ti": 2, "RTX4090": 1},
        model_name=ModelName.Qwen32B,
        # cluster
        complete_cluster_file_name="./config/single3.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol_qwen32b_3gpu_mixed_awq.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_3gpu_mixed.ini",
        real_sys_config_file_name="./config/real_sys_config.txt",
        # throughput
        duration=300,
        initial_launch_num=2,
        feeding_hwm=0.8,
        # result
        result_logging_dir="./result/qwen32b_3gpu_mixed_awq_maxflow_offline/"
    )


def qwen32b_3gpu_mixed_awq_maxflow_online():
    os.makedirs("./result/qwen32b_3gpu_mixed_awq_maxflow_online/", exist_ok=True)
    print("Running Qwen3-32B-AWQ: maxflow host + online mode (2x RTX2080Ti + 1x RTX4090)")
    run_maxflow_host_online(
        # model and machine
        machine_num_dict={"RTX2080Ti": 2, "RTX4090": 1},
        model_name=ModelName.Qwen32B,
        # cluster
        complete_cluster_file_name="./config/single3.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol_qwen32b_3gpu_mixed_awq.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_3gpu_mixed.ini",
        real_sys_config_file_name="./config/real_sys_config.txt",
        # throughput
        duration=300,
        avg_throughput=200,
        # result
        result_logging_dir="./result/qwen32b_3gpu_mixed_awq_maxflow_online/"
    )


def qwen32b_3gpu_mixed_awq_heuristic_offline(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/qwen32b_3gpu_mixed_awq_{heuristic}_offline/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host offline
    print(f"Running Qwen3-32B-AWQ: {heuristic} host + offline mode (2x RTX2080Ti + 1x RTX4090)")
    run_heuristic_host_offline(
        scheduler_name=heuristic,
        real_sys_config_file_name="./config/real_sys_config.txt",
        initial_launch_num=50,
        duration=300,
        result_logging_dir=result_dir
    )


def qwen32b_3gpu_mixed_awq_heuristic_online(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/qwen32b_3gpu_mixed_awq_{heuristic}_online/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host online
    print(f"Running Qwen3-32B-AWQ: {heuristic} host + online mode (2x RTX2080Ti + 1x RTX4090)")
    print("Configuration:")
    print("  - RTX2080Ti #1 (10.202.210.104:5001): layers 0-15")
    print("  - RTX2080Ti #2 (10.202.210.104:5002): layers 16-31")
    print("  - RTX4090 (10.130.151.13): layers 32-63")
    run_heuristic_host_online(
        scheduler_name=heuristic,
        real_sys_config_file_name="./config/real_sys_config.txt",
        avg_throughput=100,
        duration=300,
        result_logging_dir=result_dir
    )


def main():
    # parse arguments
    if len(sys.argv) != 3:
        print("Usage: python3 step2_start_host_qwen32b_3gpu_mixed_awq.py <mode> <scheduling_method>")
        print("  mode: online | offline")
        print("  scheduling_method: maxflow | swarm | random")
        print("")
        print("This script is specifically configured for Qwen3-32B-AWQ on 2x RTX2080Ti + 1x RTX4090 GPUs.")
        print("Layer distribution:")
        print("  - RTX2080Ti #1 (127.0.0.1): layers 0-15")
        print("  - RTX2080Ti #2 (127.0.0.2): layers 16-31")
        print("  - RTX4090 (10.130.151.21): layers 32-63")
        print("Quantization: AWQ (4-bit) for reduced memory footprint")
        return
    mode = sys.argv[1]
    method = sys.argv[2]

    # validate arguments
    assert mode in ["online", "offline"], f"Unsupported mode: {mode}!"
    assert method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {method}!"

    # run the appropriate configuration
    if mode == "offline":
        if method == "maxflow":
            qwen32b_3gpu_mixed_awq_maxflow_offline()
        else:
            qwen32b_3gpu_mixed_awq_heuristic_offline(method)
    else:  # online
        if method == "maxflow":
            qwen32b_3gpu_mixed_awq_maxflow_online()
        else:
            qwen32b_3gpu_mixed_awq_heuristic_online(method)


if __name__ == '__main__':
    main()
