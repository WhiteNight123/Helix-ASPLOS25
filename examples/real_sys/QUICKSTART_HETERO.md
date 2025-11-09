# Qwen3-14B异构部署快速启动指南

## 系统配置概览

```
层分配 (40层总计):
┌─────────────────┬──────────────┬───────────┬─────────┐
│ GPU             │ 类型         │ IP地址    │ 层范围  │
├─────────────────┼──────────────┼───────────┼─────────┤
│ GPU1 (主机1)    │ RTX2080Ti    │ .104/.105 │ 0-9     │
│ GPU2 (主机1)    │ RTX2080Ti    │ .104/.106 │ 10-19   │
│ GPU3 (主机2)    │ RTX4090      │ .13/.14   │ 20-29   │
│ GPU4 (主机2)    │ RTX4090      │ .13/.15   │ 30-39   │
└─────────────────┴──────────────┴───────────┴─────────┘

流水线拓扑:
Coordinator -> GPU1 -> GPU2 -> GPU3 -> GPU4 -> Coordinator
```

## 快速部署（5分钟）

### 前置条件
- [x] Docker已安装并配置NVIDIA运行时
- [x] 两台主机之间网络互通
- [x] Qwen3-14B模型文件已准备

### 1. 修改部署脚本配置

**主机1 (2080Ti) - 编辑 `deploy_2080ti.sh`:**
```bash
HELIX_PATH="/root/Helix-ASPLOS25"  # 修改为实际路径
MODEL_PATH="/data/models/qwen3-14b"  # 修改为模型路径
```

**主机2 (4090) - 编辑 `deploy_4090.sh`:**
```bash
HELIX_PATH="/root/Helix-ASPLOS25"  # 修改为实际路径
MODEL_PATH="/data/models/qwen3-14b"  # 修改为模型路径
```

### 2. 执行部署

**在主机1 (10.202.210.104) 执行:**
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
chmod +x deploy_2080ti.sh
sudo ./deploy_2080ti.sh
```

**在主机2 (10.130.151.13) 执行:**
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
chmod +x deploy_4090.sh
sudo ./deploy_4090.sh
```

### 3. 启动Coordinator

**在主机1 (10.202.210.104) 执行:**
```bash
cd /root/Helix-ASPLOS25/examples/real_sys

# 离线模式 + maxflow调度
python3 step2_start_host_qwen3_14b_hetero.py offline maxflow

# 或者在线模式
# python3 step2_start_host_qwen3_14b_hetero.py online maxflow
```

### 4. 验证部署

```bash
# 检查所有Worker容器状态
docker ps | grep helix_worker

# 查看Worker日志
docker logs -f helix_worker_gpu1_2080ti
```

## 详细部署步骤

### 步骤0: 环境准备

#### 安装Docker和NVIDIA Container Toolkit（两台主机）
```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 测试GPU访问
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

#### 克隆代码库（两台主机）
```bash
cd /root
git clone https://github.com/WhiteNight123/Helix-ASPLOS25.git
cd Helix-ASPLOS25
```

#### 准备模型文件（两台主机）
```bash
# 将Qwen3-14B模型复制到指定位置
# 确保包含以下文件:
# - config.json
# - tokenizer.json
# - tokenizer_config.json
# - model-*.safetensors

# 示例：从Hugging Face下载
# git lfs install
# git clone https://huggingface.co/Qwen/Qwen2.5-14B /data/models/qwen3-14b
```

### 步骤1: 配置网络

#### 方式1: 使用IP别名（推荐，脚本自动配置）

脚本会自动执行以下操作：

**主机1:**
```bash
sudo ip addr add 10.202.210.105/24 dev eth0
sudo ip addr add 10.202.210.106/24 dev eth0
```

**主机2:**
```bash
sudo ip addr add 10.130.151.14/24 dev eth0
sudo ip addr add 10.130.151.15/24 dev eth0
```

#### 方式2: 使用Macvlan网络（高级）

参见 `DOCKER_DEPLOYMENT_QWEN3_14B_HETERO.md` 中的详细说明。

### 步骤2: 测试网络连通性

**从主机1测试:**
```bash
ping -c 3 10.130.151.13
ping -c 3 10.130.151.14
ping -c 3 10.130.151.15
```

**从主机2测试:**
```bash
ping -c 3 10.202.210.104
ping -c 3 10.202.210.105
ping -c 3 10.202.210.106
```

如果ping不通，检查：
- 防火墙设置
- 路由配置
- 网络拓扑

### 步骤3: 构建Docker镜像

**两台主机都执行:**
```bash
cd /root/Helix-ASPLOS25
docker build -t helix:latest -f examples/real_sys/Dockerfile .
```

### 步骤4: 生成配置文件

**仅在主机1执行:**
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python3 step1_generate_system_config_qwen3_14b_2x2080ti_2x4090.py
```

这会生成 `config/real_sys_config_qwen3_14b_hetero.txt`，需要将此文件复制到主机2（如果Host在主机1上运行）。

### 步骤5: 启动Workers

**主机1 - 手动启动:**
```bash
cd /root/Helix-ASPLOS25/examples/real_sys

# GPU1
docker run -d \
  --name helix_worker_gpu1_2080ti \
  --network host \
  --gpus '"device=0"' \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /root/Helix-ASPLOS25:/workspace \
  -v /data/models/qwen3-14b:/workspace/examples/real_sys/model:ro \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.202.210.105"

# GPU2
docker run -d \
  --name helix_worker_gpu2_2080ti \
  --network host \
  --gpus '"device=1"' \
  -e CUDA_VISIBLE_DEVICES=1 \
  -v /root/Helix-ASPLOS25:/workspace \
  -v /data/models/qwen3-14b:/workspace/examples/real_sys/model:ro \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.202.210.106"
```

**主机2 - 手动启动:**
```bash
cd /root/Helix-ASPLOS25/examples/real_sys

# GPU3
docker run -d \
  --name helix_worker_gpu3_4090 \
  --network host \
  --gpus '"device=0"' \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /root/Helix-ASPLOS25:/workspace \
  -v /data/models/qwen3-14b:/workspace/examples/real_sys/model:ro \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.130.151.14"

# GPU4
docker run -d \
  --name helix_worker_gpu4_4090 \
  --network host \
  --gpus '"device=1"' \
  -e CUDA_VISIBLE_DEVICES=1 \
  -v /root/Helix-ASPLOS25:/workspace \
  -v /data/models/qwen3-14b:/workspace/examples/real_sys/model:ro \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.130.151.15"
```

### 步骤6: 启动Coordinator/Host

**在主机1执行:**
```bash
cd /root/Helix-ASPLOS25/examples/real_sys

# 离线模式（预加载请求）
python3 step2_start_host_qwen3_14b_hetero.py offline maxflow

# 或在线模式（动态请求）
# python3 step2_start_host_qwen3_14b_hetero.py online maxflow
```

## 监控和日志

### 查看容器状态
```bash
docker ps -a | grep helix
```

### 查看实时日志
```bash
# 主机1
docker logs -f helix_worker_gpu1_2080ti
docker logs -f helix_worker_gpu2_2080ti

# 主机2
docker logs -f helix_worker_gpu3_4090
docker logs -f helix_worker_gpu4_4090
```

### 监控GPU使用
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用nvtop（需要安装）
sudo apt install nvtop
nvtop
```

### 进入容器调试
```bash
docker exec -it helix_worker_gpu1_2080ti /bin/bash
```

## 停止和清理

### 停止所有Workers
```bash
# 主机1
docker stop helix_worker_gpu1_2080ti helix_worker_gpu2_2080ti

# 主机2
docker stop helix_worker_gpu3_4090 helix_worker_gpu4_4090
```

### 删除容器
```bash
# 主机1
docker rm helix_worker_gpu1_2080ti helix_worker_gpu2_2080ti

# 主机2
docker rm helix_worker_gpu3_4090 helix_worker_gpu4_4090
```

### 清理IP别名
```bash
# 主机1
sudo ip addr del 10.202.210.105/24 dev eth0
sudo ip addr del 10.202.210.106/24 dev eth0

# 主机2
sudo ip addr del 10.130.151.14/24 dev eth0
sudo ip addr del 10.130.151.15/24 dev eth0
```

## 故障排查

### 问题1: Worker无法启动
```bash
# 查看详细错误
docker logs helix_worker_gpu1_2080ti

# 检查GPU可见性
docker exec helix_worker_gpu1_2080ti nvidia-smi

# 检查模型文件
docker exec helix_worker_gpu1_2080ti ls -la /workspace/examples/real_sys/model/
```

### 问题2: Workers无法连接到Host
```bash
# 检查网络连通性
docker exec helix_worker_gpu1_2080ti ping 10.202.210.104

# 检查Host是否在监听
netstat -tulpn | grep python

# 查看Host日志
tail -f /root/Helix-ASPLOS25/examples/real_sys/log/*
```

### 问题3: 跨主机通信失败
```bash
# 测试跨主机连通性
ping 10.130.151.13
traceroute 10.130.151.13

# 检查防火墙
sudo ufw status
sudo iptables -L

# 临时关闭防火墙测试（不推荐生产环境）
sudo ufw disable
```

### 问题4: 内存不足
```bash
# 检查GPU显存
nvidia-smi

# 调整配置
# 编辑 layout/simulator_cluster_2x2080ti_2x4090.ini
# 减少 kv_cache_capacity
```

## 性能调优

### 调整调度参数

编辑 `step2_start_host_qwen3_14b_hetero.py`:

```python
# 离线模式
initial_launch_num=2,  # 增加初始请求数
feeding_hwm=0.8,       # 调整喂入高水位

# 在线模式
avg_throughput=100,    # 调整吞吐量目标
```

### 调整层分配

如果显存允许，可以调整层分配：

编辑 `layout/ilp_sol_qwen3_14b_2x2080ti_2x4090.ini`:
```ini
# 例如：让4090承担更多层
compute_node_2=[0, 1, 2, 3, 4, 5, 6, 7]        # 2080Ti: 8层
compute_node_3=[8, 9, 10, 11, 12, 13, 14, 15]  # 2080Ti: 8层
compute_node_4=[16, 17, ..., 27]                # 4090: 12层
compute_node_5=[28, 29, ..., 39]                # 4090: 12层
```

## 常用命令速查

```bash
# 查看所有Helix容器
docker ps -a | grep helix

# 重启Worker
docker restart helix_worker_gpu1_2080ti

# 查看资源使用
docker stats helix_worker_gpu1_2080ti

# 查看网络
docker network ls

# 批量停止
docker stop $(docker ps -q --filter "name=helix_worker")

# 批量删除
docker rm $(docker ps -aq --filter "name=helix_worker")
```

## 联系与支持

遇到问题请查看：
- Helix GitHub Issues
- 详细部署文档: `DOCKER_DEPLOYMENT_QWEN3_14B_HETERO.md`
- 单机部署参考: `SINGLE_MACHINE_SETUP.md`
