# 2024.11.02 Yixuan Mei
import sys

from llm_sys.worker import run_worker


def main():
    # parse arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python step3_start_worker.py <scheduling_method> [worker_ip]")
        print("  scheduling_method: maxflow | swarm | random")
        print("  worker_ip: (optional) IP address for this worker (e.g., 127.0.0.1, 127.0.0.2, etc.)")
        return
    scheduling_method = sys.argv[1]
    worker_ip = sys.argv[2] if len(sys.argv) == 3 else None

    # check arguments
    assert scheduling_method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {scheduling_method}!"
    print(f"Starting worker with scheduling method: {scheduling_method}.")
    if worker_ip:
        print(f"Using specified worker IP: {worker_ip}")

    # run worker
    run_worker(scheduling_method=scheduling_method, model_name="./model", worker_ip=worker_ip)


if __name__ == '__main__':
    main()
