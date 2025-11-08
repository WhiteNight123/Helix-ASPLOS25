# 2024.10.27 Start Host for Qwen3-32B with 6x RTX2080Ti
import os
import sys

from llm_sys.maxflow_host import run_maxflow_host_online, run_maxflow_host_offline
from llm_sys.heuristic_host import run_heuristic_host_online, run_heuristic_host_offline
from simulator.event_simulator.cluster_simulator import ModelName


def qwen32b_maxflow_offline():
    os.makedirs("./result/qwen32b_maxflow_offline/", exist_ok=True)
    print("Running Qwen3-32B: maxflow host + offline mode")
    run_maxflow_host_offline(
        # model and machine
        machine_num_dict={"RTX2080Ti": 7},
        model_name=ModelName.Qwen32B,
        # cluster
        complete_cluster_file_name="./config/single7.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol_qwen32b_7gpu.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_7gpu.ini",
        real_sys_config_file_name="./config/real_sys_config.txt",
        # throughput
        duration=300,
        initial_launch_num=2,
        feeding_hwm=0.8,
        # result
        result_logging_dir="./result/qwen32b_maxflow_offline/"
    )


def qwen32b_maxflow_online():
    os.makedirs("./result/qwen32b_maxflow_online/", exist_ok=True)
    print("Running Qwen3-32B: maxflow host + online mode")
    run_maxflow_host_online(
        # model and machine
        machine_num_dict={"RTX2080Ti": 7},
        model_name=ModelName.Qwen32B,
        # cluster
        complete_cluster_file_name="./config/single7.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol_qwen32b_7gpu.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_7gpu.ini",
        real_sys_config_file_name="./config/real_sys_config.txt",
        # throughput
        duration=300,
        avg_throughput=200,
        # result
        result_logging_dir="./result/qwen32b_maxflow_online/"
    )


def qwen32b_heuristic_offline(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/qwen32b_{heuristic}_offline/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host offline
    print(f"Running Qwen3-32B: {heuristic} host + offline mode")
    run_heuristic_host_offline(
        scheduler_name=heuristic,
        real_sys_config_file_name="./config/real_sys_config.txt",
        initial_launch_num=50,
        duration=300,
        result_logging_dir=result_dir
    )


def qwen32b_heuristic_online(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/qwen32b_{heuristic}_online/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host online
    print(f"Running Qwen3-32B: {heuristic} host + online mode")
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
        print("Usage: python3 step2_start_host_qwen32b.py <mode> <scheduling_method>")
        print("  mode: online | offline")
        print("  scheduling_method: maxflow | swarm | random")
        print("")
        print("This script is specifically configured for Qwen3-32B on 7x RTX2080Ti GPUs.")
        print("Layer distribution: 9-10 layers per GPU (64 layers total)")
        return
    mode = sys.argv[1]
    method = sys.argv[2]

    # check arguments
    assert mode in ["online", "offline"], f"Unsupported mode: {mode}!"
    assert method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {method}!"

    # run the corresponding function
    if mode == "offline":
        if method == "maxflow":
            qwen32b_maxflow_offline()
        else:
            qwen32b_heuristic_offline(method)
    else:  # online
        if method == "maxflow":
            qwen32b_maxflow_online()
        else:
            qwen32b_heuristic_online(method)


if __name__ == '__main__':
    main()
