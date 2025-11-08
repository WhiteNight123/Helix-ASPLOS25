from llm_sys.gen_sys_config import gen_sys_config
from simulator.event_simulator.cluster_simulator import ModelName


def main():
    gen_sys_config(
        host_ip="10.202.210.104",
        type2ips={"RTX2080Ti": ["127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4", "127.0.0.5", "127.0.0.6", "127.0.0.7"]},
        # model and machine
        machine_num_dict={"RTX2080Ti": 7},
        model_name=ModelName.Qwen32B,
        # cluster
        complete_cluster_file_name="./config/single7.ini",
        machine_profile_file_name="./config/machine_profile.ini",
        # model placement
        solution_file_name="./layout/ilp_sol_qwen32b_7gpu.ini",
        simulator_cluster_file_name="./layout/simulator_cluster_7gpu.ini",
        # output directory
        output_dir="./config",
        output_file_name="real_sys_config.txt"
    )
    print("System config generated to ./config/real_sys_config.txt!")
    print("Configuration for 7x RTX2080Ti GPUs running Qwen3-32B (64 layers)")
    print("Layer distribution: 9-10 layers per GPU")


if __name__ == '__main__':
    main()
