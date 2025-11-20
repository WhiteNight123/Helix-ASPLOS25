#!/usr/bin/env python3
# Start Host for Qwen3-14B with heterogeneous GPUs (2x RTX2080Ti + 2x RTX4090)
import os
import sys

from llm_sys.maxflow_host import run_maxflow_host_online, run_maxflow_host_offline
from llm_sys.heuristic_host import run_heuristic_host_online, run_heuristic_host_offline
from simulator.event_simulator.cluster_simulator import ModelName


def qwen3_14b_hetero_maxflow_offline():
    os.makedirs("./result/qwen3_14b_hetero_maxflow_offline/", exist_ok=True)
    print("Running Qwen3-14B (2x2080Ti + 2x4090): maxflow host + offline mode")
    run_maxflow_host_offline(
        # model and machine
        machine_num_dict={"RTX2080Ti": 2, "RTX4090": 2},
        model_name=ModelName.Qwen3_14B,
        # cluster
        complete_cluster_file_name="./config/heterogeneous_2x2080ti_2x4090.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol_qwen3_14b_2x2080ti_2x4090.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_2x2080ti_2x4090.ini",
        real_sys_config_file_name="./config/real_sys_config_qwen3_14b_hetero.txt",
        # throughput
        duration=300,
        initial_launch_num=2,
        feeding_hwm=0.8,
        # result
        result_logging_dir="./result/qwen3_14b_hetero_maxflow_offline/"
    )


def qwen3_14b_hetero_maxflow_online():
    os.makedirs("./result/qwen3_14b_hetero_maxflow_online/", exist_ok=True)
    print("Running Qwen3-14B (2x2080Ti + 2x4090): maxflow host + online mode")
    run_maxflow_host_online(
        # model and machine
        machine_num_dict={"RTX2080Ti": 2, "RTX4090": 2},
        model_name=ModelName.Qwen3_14B,
        # cluster
        complete_cluster_file_name="./config/heterogeneous_2x2080ti_2x4090.ini",
        machine_profile_name="./config/machine_profile.ini",
        # solution
        solution_file_name="./layout/ilp_sol_qwen3_14b_2x2080ti_2x4090.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_2x2080ti_2x4090.ini",
        real_sys_config_file_name="./config/real_sys_config_qwen3_14b_hetero.txt",
        # throughput
        duration=60,
        avg_rps=1,
        # result
        result_logging_dir="./result/qwen3_14b_hetero_maxflow_online/"
    )


def qwen3_14b_hetero_heuristic_offline(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/qwen3_14b_hetero_{heuristic}_offline/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host offline
    print(f"Running Qwen3-14B (2x2080Ti + 2x4090): {heuristic} host + offline mode")
    run_heuristic_host_offline(
        scheduler_name=heuristic,
        real_sys_config_file_name="./config/real_sys_config_qwen3_14b_hetero.txt",
        initial_launch_num=50,
        duration=300,
        result_logging_dir=result_dir
    )


def qwen3_14b_hetero_heuristic_online(heuristic: str):
    # check arguments and create result directory
    assert heuristic in ["swarm", "random"], f"Unsupported heuristic: {heuristic}!"
    result_dir = f"./result/qwen3_14b_hetero_{heuristic}_online/"
    os.makedirs(result_dir, exist_ok=True)

    # run heuristic host online
    print(f"Running Qwen3-14B (2x2080Ti + 2x4090): {heuristic} host + online mode")
    run_heuristic_host_online(
        scheduler_name=heuristic,
        real_sys_config_file_name="./config/real_sys_config_qwen3_14b_hetero.txt",
        avg_throughput=50,
        duration=300,
        result_logging_dir=result_dir
    )


def main():
    # parse arguments
    if len(sys.argv) != 3:
        print("Usage: python3 step2_start_host_qwen3_14b_hetero.py <mode> <scheduling_method>")
        print("  mode: online | offline")
        print("  scheduling_method: maxflow | swarm | random")
        print("")
        print("This script is configured for Qwen3-14B on 2x RTX2080Ti + 2x RTX4090 GPUs.")
        print("")
        print("Example:")
        print("  python3 step2_start_host_qwen3_14b_hetero.py offline maxflow")
        return
    mode = sys.argv[1]
    method = sys.argv[2]

    # validate arguments
    assert mode in ["online", "offline"], f"Unsupported mode: {mode}!"
    assert method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {method}!"

    # run the corresponding function
    if mode == "offline":
        if method == "maxflow":
            qwen3_14b_hetero_maxflow_offline()
        else:
            qwen3_14b_hetero_heuristic_offline(method)
    else:  # online
        if method == "maxflow":
            qwen3_14b_hetero_maxflow_online()
        else:
            qwen3_14b_hetero_heuristic_online(method)


if __name__ == '__main__':
    main()
