# 2024.11.06 Start Host for Qwen3-14B with 4x RTX2080Ti
import os
import sys

from llm_sys.maxflow_host import run_maxflow_host_online, run_maxflow_host_offline
from llm_sys.heuristic_host import run_heuristic_host_online, run_heuristic_host_offline
from simulator.event_simulator.cluster_simulator import ModelName


def qwen3_14b_maxflow_offline():
    os.makedirs("./result/qwen3_14b_4gpu_maxflow_offline/", exist_ok=True)
    print("Running Qwen3-14B (4 GPUs): maxflow host + offline mode")
    run_maxflow_host_offline(
        # model and machine
        machine_num_dict={"RTX2080Ti": 4},
        model_name=ModelName.Qwen3_14B,
        # cluster
        complete_cluster_file_name="./config/single4.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol_qwen3_14b_4gpu.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_4gpu.ini",
        real_sys_config_file_name="./config/real_sys_config.txt",
        # throughput
        duration=300,
        initial_launch_num=2,
        feeding_hwm=0.8,
        # result
        result_logging_dir="./result/qwen3_14b_4gpu_maxflow_offline/"
    )


def qwen3_14b_maxflow_online():
    os.makedirs("./result/qwen3_14b_4gpu_maxflow_online/", exist_ok=True)
    print("Running Qwen3-14B (4 GPUs): maxflow host + online mode")
    run_maxflow_host_online(
        # model and machine
        machine_num_dict={"RTX2080Ti": 4},
        model_name=ModelName.Qwen3_14B,
        # cluster
        complete_cluster_file_name="./config/single4.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol_qwen3_14b_4gpu.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_4gpu.ini",
        real_sys_config_file_name="./config/real_sys_config.txt",
        # throughput
        duration=300,
        avg_rps=3,
        # result
        result_logging_dir="./result/qwen3_14b_4gpu_maxflow_online/"
    )


def qwen3_14b_heuristic_offline(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/qwen3_14b_4gpu_{heuristic}_offline/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host offline
    print(f"Running Qwen3-14B (4 GPUs): {heuristic} host + offline mode")
    run_heuristic_host_offline(
        scheduler_name=heuristic,
        real_sys_config_file_name="./config/real_sys_config.txt",
        initial_launch_num=50,
        duration=300,
        result_logging_dir=result_dir
    )


def qwen3_14b_heuristic_online(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/qwen3_14b_4gpu_{heuristic}_online/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host online
    print(f"Running Qwen3-14B (4 GPUs): {heuristic} host + online mode")
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
        print("Usage: python3 step2_start_host_qwen3_14b_4gpu.py <mode> <scheduling_method>")
        print("  mode: online | offline")
        print("  scheduling_method: maxflow | swarm | random")
        print("")
        print("This script is specifically configured for Qwen3-14B on 4x RTX2080Ti GPUs.")
        return
    mode = sys.argv[1]
    method = sys.argv[2]

    # call the corresponding function
    if mode == "offline":
        if method == "maxflow":
            qwen3_14b_maxflow_offline()
        elif method in ["swarm", "random"]:
            qwen3_14b_heuristic_offline(method)
        else:
            print(f"Unknown scheduling method: {method}!")
    elif mode == "online":
        if method == "maxflow":
            qwen3_14b_maxflow_online()
        elif method in ["swarm", "random"]:
            qwen3_14b_heuristic_online(method)
        else:
            print(f"Unknown scheduling method: {method}!")
    else:
        print(f"Unknown mode: {mode}!")


if __name__ == '__main__':
    main()
