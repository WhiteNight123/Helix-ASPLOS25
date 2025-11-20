#!/bin/bash
set -e  # 出错时退出

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
LOG_DIR_CONTAINER="/Helix-ASPLOS25/logs"

mkdir -p "$LOG_DIR"
echo "Worker logs will be stored under $LOG_DIR"

echo "Removing existing containers..."
docker rm -f helix_worker_gpu1_2080ti helix_worker_gpu2_2080ti helix_worker_gpu3_2080ti helix_worker_gpu4_2080ti helix_coordinator 2>/dev/null || true

echo "Starting worker containers..."
for i in {1..4}; do
  gpu_id=$((i-1))
  ip="10.0.1.$((10 + i))"
  name="helix_worker_gpu${i}_2080ti"
  echo "Launching $name on GPU $gpu_id with IP $ip"
  log_file_container="$LOG_DIR_CONTAINER/${name}.log"
  log_file_host="$LOG_DIR/${name}.log"
  : > "$log_file_host"

  docker run -d \
    --name "$name" \
    --network test_heter \
    --ip "$ip" \
    --gpus "device=$gpu_id" \
    -e HELIX_HOST_IP=10.0.1.10 \
    -e VLLM_LOG_LEVEL=debug \
    -v "$REPO_ROOT":/Helix-ASPLOS25 \
    myhelix:latest \
    bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow $ip > $log_file_container 2>&1"

  echo "Logs for $name -> $log_file_host"
done

echo "Starting coordinator..."
docker run -it --rm \
  --name helix_coordinator \
  --network test_heter \
  --ip 10.0.1.10 \
  --gpus all \
  -e HELIX_HOST_IP=10.0.1.10 \
  -v /mnt/lvm-data/home/dataset/sharegpt:/data \
  -v "$REPO_ROOT":/Helix-ASPLOS25 \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step2_start_host_qwen3_14b_hetero.py online maxflow"