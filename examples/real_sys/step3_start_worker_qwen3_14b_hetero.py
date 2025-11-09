#!/usr/bin/env python3
# Start Worker for Qwen3-14B on heterogeneous GPUs (2x RTX2080Ti + 2x RTX4090)
import sys

from llm_sys.worker import run_worker


def main():
    # parse arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 step3_start_worker_qwen3_14b_hetero.py <scheduling_method> [worker_ip]")
        print("  scheduling_method: maxflow | swarm | random")
        print("  worker_ip: (optional) IP address for this worker")
        print("")
        print("IP addresses for heterogeneous setup:")
        print("  - GPU1 (2080Ti): 10.202.210.105")
        print("  - GPU2 (2080Ti): 10.202.210.106")
        print("  - GPU3 (4090):   10.130.151.14")
        print("  - GPU4 (4090):   10.130.151.15")
        print("")
        print("Example:")
        print("  python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.202.210.105")
        return
    scheduling_method = sys.argv[1]
    worker_ip = sys.argv[2] if len(sys.argv) == 3 else None

    # check arguments
    assert scheduling_method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {scheduling_method}!"
    print(f"Starting Qwen3-14B worker (heterogeneous setup) with scheduling method: {scheduling_method}.")
    if worker_ip:
        print(f"Using specified worker IP: {worker_ip}")
    else:
        print("WARNING: No worker IP specified. Worker will try to auto-detect IP.")

    # run worker with Qwen3-14B model
    run_worker(scheduling_method=scheduling_method, model_name="./model", worker_ip=worker_ip)


if __name__ == '__main__':
    main()
