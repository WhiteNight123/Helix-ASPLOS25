#!/usr/bin/env python3
# Generate system config for Qwen3-32B-AWQ on 4x RTX2080Ti GPUs
from llm_sys.gen_sys_config import gen_sys_config
from simulator.event_simulator.cluster_simulator import ModelName


def main():
    gen_sys_config(
        host_ip="10.202.210.104",  # Update this to your host IP
        type2ips={"RTX2080Ti": ["127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4"]},
        # model and machine
        machine_num_dict={"RTX2080Ti": 4},
        model_name=ModelName.Qwen32B,
        # cluster
        complete_cluster_file_name="./config/single4_awq.ini",
        machine_profile_file_name="./config/machine_profile.ini",
        # model placement
        solution_file_name="./layout/ilp_sol_qwen32b_4gpu_awq.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_qwen32b_4gpu_awq.ini",
        # output directory
        output_dir="./config",
        output_file_name="real_sys_config_qwen32b_awq.txt"
    )
    print("System config generated to ./config/real_sys_config_qwen32b_awq.txt!")
    print("Configuration for 4x RTX2080Ti GPUs running Qwen3-32B-AWQ (64 layers)")
    print("Layer distribution: 16 layers per GPU")
    print("Quantization: AWQ (4-bit) for reduced memory footprint")
    print("")
    print("Note: Make sure your model directory contains AWQ quantized weights")


if __name__ == '__main__':
    main()
