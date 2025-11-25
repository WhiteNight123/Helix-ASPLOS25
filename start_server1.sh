#!/bin/bash

# Script to start Helix coordinator and 2080Ti workers on Server 1
# This script should be run on the primary server (server1)

set -e

echo "======================================"
echo "Starting Helix on Server 1"
echo "Time: $(date)"
echo "======================================"

# Create log directory if it doesn't exist
LOG_DIR="/root/Helix-ASPLOS25/examples/real_sys/log"
mkdir -p "$LOG_DIR"

# Clean up existing containers if they exist
echo "[$(date)] Cleaning up existing containers..."
docker rm -f helix_worker_gpu1_2080ti 2>/dev/null || true
docker rm -f helix_worker_gpu2_2080ti 2>/dev/null || true
docker rm -f helix_coordinator 2>/dev/null || true
echo "[$(date)] Cleanup completed"

# Step 1: Start Worker containers for 2080Ti GPUs
echo "[$(date)] Starting Worker 1 (2080Ti GPU0)..."
docker run -d \
  --name helix_worker_gpu1_2080ti \
  --network test_heter \
  --ip 10.0.1.11 \
  --gpus '"device=0"' \
  -e HELIX_HOST_IP=10.0.1.10 \
  -e VLLM_LOG_LEVEL=debug \
  -v /root/Helix-ASPLOS25:/Helix-ASPLOS25 \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.0.1.11 2>&1 | tee log/worker_gpu1_2080ti.log"

echo "[$(date)] Worker 1 started (Container ID: $(docker ps -q -f name=helix_worker_gpu1_2080ti))"

echo "[$(date)] Starting Worker 2 (2080Ti GPU1)..."
docker run -d \
  --name helix_worker_gpu2_2080ti \
  --network test_heter \
  --ip 10.0.1.12 \
  --gpus '"device=1"' \
  -e HELIX_HOST_IP=10.0.1.10 \
  -e VLLM_LOG_LEVEL=debug \
  -v /root/Helix-ASPLOS25:/Helix-ASPLOS25 \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.0.1.12 2>&1 | tee log/worker_gpu2_2080ti.log"

echo "[$(date)] Worker 2 started (Container ID: $(docker ps -q -f name=helix_worker_gpu2_2080ti))"

# Wait a bit for workers to initialize
echo "[$(date)] Waiting 10 seconds for workers to initialize..."
sleep 10

# Step 2: Start Coordinator
echo "[$(date)] Starting Coordinator..."
echo "======================================"
echo "Coordinator is starting in interactive mode"
echo "Logs will be displayed below"
echo "======================================"

docker run -it --rm \
  --name helix_coordinator \
  --network test_heter \
  --ip 10.0.1.10 \
  --gpus all \
  -e HELIX_HOST_IP=10.0.1.10 \
  -v /mnt/lvm-data/home/dataset/sharegpt:/data \
  -v /root/Helix-ASPLOS25:/Helix-ASPLOS25 \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step2_start_host_qwen3_14b_hetero.py online maxflow 2>&1 | tee log/coordinator.log"

echo "[$(date)] Coordinator stopped"
echo "======================================"
echo "Server 1 completed at $(date)"
echo "======================================"
