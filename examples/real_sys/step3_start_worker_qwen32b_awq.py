#!/usr/bin/env python3
# Start Worker for Qwen3-32B-AWQ on RTX2080Ti
import sys

from llm_sys.worker import run_worker


def main():
    # parse arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python step3_start_worker_qwen32b_awq.py <scheduling_method> [worker_ip]")
        print("  scheduling_method: maxflow | swarm | random")
        print("  worker_ip: (optional) IP address for this worker (e.g., 127.0.0.1, 127.0.0.2, etc.)")
        print("")
        print("This worker is configured for Qwen3-32B-AWQ with:")
        print("  - AWQ 4-bit quantization")
        print("  - Optimized for 4x RTX2080Ti GPUs")
        print("  - 16 layers per GPU (64 layers total)")
        return
    scheduling_method = sys.argv[1]
    worker_ip = sys.argv[2] if len(sys.argv) == 3 else None

    # check arguments
    assert scheduling_method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {scheduling_method}!"
    print(f"Starting Qwen3-32B-AWQ worker with scheduling method: {scheduling_method}.")
    if worker_ip:
        print(f"Using specified worker IP: {worker_ip}")
    print("Quantization: AWQ (4-bit)")

    # run worker with Qwen3-32B-AWQ model
    # Using 0.9 VRAM usage since AWQ quantization uses less memory
    run_worker(
        scheduling_method=scheduling_method,
        model_name="./model",
        worker_ip=worker_ip,
        vram_usage=0.9,  # Can use more VRAM with AWQ quantization
        quantization="awq"
    )


if __name__ == '__main__':
    main()
