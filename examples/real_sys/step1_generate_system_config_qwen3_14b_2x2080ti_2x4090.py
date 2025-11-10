from llm_sys.gen_sys_config import gen_sys_config
from simulator.event_simulator.cluster_simulator import ModelName


def main():
    """
    Generate system config for Qwen3-14B with heterogeneous GPUs:
    - 2x RTX2080Ti (10.202.210.104 - Docker, each GPU needs separate IP)
    - 2x RTX4090 (10.130.151.13 - Docker, each GPU needs separate IP)
    
    Layer distribution (40 layers total):
    - GPU1 (2080Ti): layers 0-9   (10 layers)
    - GPU2 (2080Ti): layers 10-19 (10 layers)
    - GPU3 (4090):   layers 20-29 (10 layers)
    - GPU4 (4090):   layers 30-39 (10 layers)
    
    Pipeline topology: GPU1 -> GPU2 -> GPU3 -> GPU4
    
    Since Helix requires one IP per GPU and doesn't support ports,
    Docker containers need to use host network mode or macvlan.
    We'll use different IPs for each GPU container.
    """
    gen_sys_config(
        # Host IP (coordinator node)
        host_ip="10.100.0.1",
        
        # Worker IPs - each GPU needs a unique IP
        # In Docker, you can use macvlan or host network with different IP aliases
        type2ips={
            "RTX2080Ti": [
                "10.100.0.11",  # GPU1 on first 2080Ti host
                "10.100.0.12",  # GPU2 on first 2080Ti host
            ],
            "RTX4090": [
                "10.100.0.13",   # GPU1 on 4090 host
                "10.100.0.14",   # GPU2 on 4090 host
            ]
        },
        
        # Machine configuration
        machine_num_dict={"RTX2080Ti": 2, "RTX4090": 2},
        model_name=ModelName.Qwen3_14B,
        
        # Cluster configuration files
        complete_cluster_file_name="./config/heterogeneous_2x2080ti_2x4090.ini",
        machine_profile_file_name="./config/machine_profile.ini",
        
        # Model placement solution
        solution_file_name="./layout/ilp_sol_qwen3_14b_2x2080ti_2x4090.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_2x2080ti_2x4090.ini",
        
        # Output directory
        output_dir="./config",
        output_file_name="real_sys_config_qwen3_14b_hetero.txt"
    )
    print("System config generated to ./config/real_sys_config_qwen3_14b_hetero.txt!")
    print("")
    print("NOTE: Docker deployment requirements:")
    print("1. Each GPU container needs a unique IP address")
    print("2. Use Docker macvlan network or host network with IP aliases")
    print("3. Ensure network connectivity between all IPs")
    print("")
    print("IP allocation:")
    print("  - Host/Coordinator: 10.202.210.104")
    print("  - GPU1 (2080Ti): 10.202.210.105")
    print("  - GPU2 (2080Ti): 10.202.210.106")
    print("  - GPU3 (4090): 10.130.151.14")
    print("  - GPU4 (4090): 10.130.151.15")


if __name__ == '__main__':
    main()
