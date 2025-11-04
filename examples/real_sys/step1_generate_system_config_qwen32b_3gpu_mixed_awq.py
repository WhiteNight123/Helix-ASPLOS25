#!/usr/bin/env python3
# Generate system config for Qwen3-32B-AWQ on 2x RTX2080Ti + 1x RTX4090 GPUs
from llm_sys.gen_sys_config import gen_sys_config
from simulator.event_simulator.cluster_simulator import ModelName


def main():
    gen_sys_config(
        host_ip="10.202.210.104",  # Update this to your host IP
        type2ips={
            "RTX2080Ti": ["127.0.0.1", "127.0.0.2"],
            "RTX4090": ["10.130.151.21"]
        },
        # model and machine
        machine_num_dict={"RTX2080Ti": 2, "RTX4090": 1},
        model_name=ModelName.Qwen32B,
        # cluster
        complete_cluster_file_name="./config/single3_mixed_awq.ini",
        machine_profile_file_name="./config/machine_profile.ini",
        # model placement
        solution_file_name="./layout/ilp_sol_qwen32b_3gpu_mixed_awq.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_qwen32b_3gpu_mixed_awq.ini",
        # output directory
        output_dir="./config",
        output_file_name="real_sys_config_qwen32b_3gpu_mixed_awq.txt"
    )
    print("System config generated to ./config/real_sys_config_qwen32b_3gpu_mixed_awq.txt!")
    print("Configuration for 2x RTX2080Ti + 1x RTX4090 GPUs running Qwen3-32B-AWQ (64 layers)")
    print("Layer distribution:")
    print("  - RTX2080Ti #1 (127.0.0.1): layers 0-15")
    print("  - RTX2080Ti #2 (127.0.0.2): layers 16-31")
    print("  - RTX4090 (10.130.151.21): layers 32-63")
    print("Quantization: AWQ (4-bit) for reduced memory footprint")
    print("")
    print("Note: Make sure your model directory contains AWQ quantized weights")


if __name__ == '__main__':
    main()
