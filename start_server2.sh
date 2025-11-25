#!/bin/bash

# Script to start 4090 workers on Server 2
# This script should be run on the secondary server (server2)

set -e

echo "======================================"
echo "Starting Helix Workers on Server 2"
echo "Time: $(date)"
echo "======================================"

# Create log directory if it doesn't exist
LOG_DIR="/home/emnets-2/gxq/Helix-ASPLOS25/examples/real_sys/log"
mkdir -p "$LOG_DIR"

# Clean up existing containers if they exist
echo "[$(date)] Cleaning up existing containers..."
docker rm -f helix_worker_gpu3_4090 2>/dev/null || true
docker rm -f helix_worker_gpu4_4090 2>/dev/null || true
echo "[$(date)] Cleanup completed"

# Step 1: Start Worker 3 (4090 GPU2)
echo "[$(date)] Starting Worker 3 (4090 GPU2)..."
docker run -d \
  --name helix_worker_gpu3_4090 \
  --network test_heter \
  --ip 10.0.1.13 \
  --gpus '"device=2"' \
  -e HELIX_HOST_IP=10.0.1.10 \
  -e VLLM_LOG_LEVEL=debug \
  -v /home/emnets-2/gxq/Helix-ASPLOS25:/Helix-ASPLOS25 \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.0.1.13 2>&1 | tee log/worker_gpu3_4090.log"

echo "[$(date)] Worker 3 started (Container ID: $(docker ps -q -f name=helix_worker_gpu3_4090))"

# Step 2: Start Worker 4 (4090 GPU3)
echo "[$(date)] Starting Worker 4 (4090 GPU3)..."
docker run -d \
  --name helix_worker_gpu4_4090 \
  --network test_heter \
  --ip 10.0.1.14 \
  --gpus '"device=3"' \
  -e HELIX_HOST_IP=10.0.1.10 \
  -e VLLM_LOG_LEVEL=debug \
  -v /home/emnets-2/gxq/Helix-ASPLOS25:/Helix-ASPLOS25 \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.0.1.14 2>&1 | tee log/worker_gpu4_4090.log"

echo "[$(date)] Worker 4 started (Container ID: $(docker ps -q -f name=helix_worker_gpu4_4090))"

echo "======================================"
echo "All workers on Server 2 started successfully"
echo "Time: $(date)"
echo "======================================"
echo ""
echo "To check worker status:"
echo "  docker ps -f name=helix_worker"
echo ""
echo "To view worker logs:"
echo "  docker logs -f helix_worker_gpu3_4090"
echo "  docker logs -f helix_worker_gpu4_4090"
echo ""
echo "Or check log files at:"
echo "  $LOG_DIR/worker_gpu3_4090.log"
echo "  $LOG_DIR/worker_gpu4_4090.log"
echo "======================================"
