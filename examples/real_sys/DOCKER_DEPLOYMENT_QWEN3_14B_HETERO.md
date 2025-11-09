# Qwen3-14B异构GPU部署指南（2x RTX2080Ti + 2x RTX4090）

## 系统架构

### 硬件配置
- **主机1 (10.202.210.104)**: 2x RTX2080Ti
  - GPU1: 10.202.210.105 - 推理层0-9 (10层)
  - GPU2: 10.202.210.106 - 推理层10-19 (10层)
  
- **主机2 (10.130.151.13)**: 2x RTX4090
  - GPU3: 10.130.151.14 - 推理层20-29 (10层)
  - GPU4: 10.130.151.15 - 推理层30-39 (10层)

### 流水线拓扑
```
Coordinator (10.202.210.104) -> GPU1 (2080Ti) -> GPU2 (2080Ti) -> GPU3 (4090) -> GPU4 (4090) -> Coordinator
```

## Docker网络配置

### 重要说明
由于Helix的限制：
1. **每个GPU需要独立的IP地址**
2. **不支持端口映射**
3. **相邻节点必须能互相访问**

因此需要使用Docker的macvlan网络或host网络模式。

### 方案1: Macvlan网络（推荐）

#### 主机1 (10.202.210.104) - 2080Ti
```bash
# 创建macvlan网络
docker network create -d macvlan \
  --subnet=10.202.210.0/24 \
  --gateway=10.202.210.1 \
  -o parent=enp97s0f0 \
  helix_network_2080ti

# 启动GPU1容器 (10.202.210.105)
docker run -d \
  --name helix_worker_gpu1_2080ti \
  --network helix_network_2080ti \
  --ip 10.202.210.105 \
  --gpus '"device=5"' \
  -v /path/to/Helix-ASPLOS25:/workspace \
  myhelix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.202.210.105"

# 启动GPU2容器 (10.202.210.106)
docker run -d \
  --name helix_worker_gpu2_2080ti \
  --network helix_network_2080ti \
  --ip 10.202.210.106 \
  --gpus '"device=6"' \
  -v /path/to/Helix-ASPLOS25:/workspace \
  -v /path/to/model:/workspace/examples/real_sys/model \
  myhelix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.202.210.106"
```

#### 主机2 (10.130.151.13) - 4090
```bash
# 创建macvlan网络
docker network create -d macvlan \
  --subnet=10.130.151.0/24 \
  --gateway=10.130.151.1 \
  -o parent=eth0 \
  helix_network_4090

# 启动GPU3容器 (10.130.151.14)
docker run -d \
  --name helix_worker_gpu3_4090 \
  --network helix_network_4090 \
  --ip 10.130.151.14 \
  --gpus '"device=0"' \
  -v /path/to/Helix-ASPLOS25:/workspace \
  -v /path/to/model:/workspace/examples/real_sys/model \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.130.151.14"

# 启动GPU4容器 (10.130.151.15)
docker run -d \
  --name helix_worker_gpu4_4090 \
  --network helix_network_4090 \
  --ip 10.130.151.15 \
  --gpus '"device=1"' \
  -v /path/to/Helix-ASPLOS25:/workspace \
  -v /path/to/model:/workspace/examples/real_sys/model \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.130.151.15"
```

### 方案2: IP别名（适用于host网络）

#### 主机1 (10.202.210.104) - 2080Ti
```bash
# 添加IP别名
sudo ip addr add 10.202.210.105/24 dev eth0
sudo ip addr add 10.202.210.106/24 dev eth0

# 启动容器使用host网络
docker run -d \
  --name helix_worker_gpu1_2080ti \
  --network host \
  --gpus '"device=0"' \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /path/to/Helix-ASPLOS25:/workspace \
  -v /path/to/model:/workspace/examples/real_sys/model \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.202.210.105"

docker run -d \
  --name helix_worker_gpu2_2080ti \
  --network host \
  --gpus '"device=1"' \
  -e CUDA_VISIBLE_DEVICES=1 \
  -v /path/to/Helix-ASPLOS25:/workspace \
  -v /path/to/model:/workspace/examples/real_sys/model \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.202.210.106"
```

#### 主机2 (10.130.151.13) - 4090
```bash
# 添加IP别名
sudo ip addr add 10.130.151.14/24 dev eth0
sudo ip addr add 10.130.151.15/24 dev eth0

# 启动容器
docker run -d \
  --name helix_worker_gpu3_4090 \
  --network host \
  --gpus '"device=0"' \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /path/to/Helix-ASPLOS25:/workspace \
  -v /path/to/model:/workspace/examples/real_sys/model \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.130.151.14"

docker run -d \
  --name helix_worker_gpu4_4090 \
  --network host \
  --gpus '"device=1"' \
  -e CUDA_VISIBLE_DEVICES=1 \
  -v /path/to/Helix-ASPLOS25:/workspace \
  -v /path/to/model:/workspace/examples/real_sys/model \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.130.151.15"
```

## 部署步骤

### 步骤1: 构建Docker镜像（两台主机都需要）

```bash
cd /path/to/Helix-ASPLOS25
docker build -t helix:latest .
```

如果没有Dockerfile，创建一个：
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
RUN pip3 install -e .

CMD ["/bin/bash"]
```

### 步骤2: 准备模型文件

确保模型文件在 `./model/` 目录下，包括：
- config.json
- tokenizer.json
- tokenizer_config.json
- 模型权重文件

### 步骤3: 生成系统配置（在主机1上执行）

```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python3 step1_generate_system_config_qwen3_14b_2x2080ti_2x4090.py
```

这将生成 `./config/real_sys_config_qwen3_14b_hetero.txt`

### 步骤4: 启动Coordinator（在主机1上执行）

```bash
# 可以在宿主机直接运行，或在容器中运行
python3 step2_start_host_qwen3_14b_hetero.py offline maxflow
```

或使用Docker：
```bash
docker run -it --rm \
  --network host \
  -v /path/to/Helix-ASPLOS25:/workspace \
  helix:latest \
  bash -c "cd /workspace/examples/real_sys && python3 step2_start_host_qwen3_14b_hetero.py offline maxflow"
```

### 步骤5: 启动Workers

按照上述Docker命令在两台主机上启动4个worker容器。

### 步骤6: 验证连接

```bash
# 检查各容器状态
docker ps

# 查看日志
docker logs helix_worker_gpu1_2080ti
docker logs helix_worker_gpu2_2080ti
docker logs helix_worker_gpu3_4090
docker logs helix_worker_gpu4_4090

# 测试网络连通性
# 从GPU1容器测试到其他GPU
docker exec helix_worker_gpu1_2080ti ping -c 3 10.202.210.106
docker exec helix_worker_gpu1_2080ti ping -c 3 10.130.151.14
docker exec helix_worker_gpu1_2080ti ping -c 3 10.130.151.15
```

## 网络配置要点

### 防火墙设置
确保两台主机之间的防火墙允许通信：
```bash
# 在两台主机上执行
sudo ufw allow from 10.202.210.0/24
sudo ufw allow from 10.130.151.0/24
```

### 路由配置
如果两台主机在不同网段，需要配置路由：

主机1 (10.202.210.104):
```bash
sudo ip route add 10.130.151.0/24 via <gateway_ip>
```

主机2 (10.130.151.13):
```bash
sudo ip route add 10.202.210.0/24 via <gateway_ip>
```

### 测试连通性
```bash
# 从主机1测试到主机2
ping 10.130.151.13
ping 10.130.151.14
ping 10.130.151.15

# 从主机2测试到主机1
ping 10.202.210.104
ping 10.202.210.105
ping 10.202.210.106
```

## 故障排查

### 问题1: Worker无法连接到Coordinator
- 检查IP地址配置是否正确
- 验证网络连通性
- 检查防火墙设置
- 查看容器日志

### 问题2: GPU无法识别
```bash
# 检查宿主机GPU
nvidia-smi

# 检查容器内GPU
docker exec helix_worker_gpu1_2080ti nvidia-smi
```

### 问题3: 内存不足
- 检查VRAM使用情况
- 调整KV cache配置
- 减少batch size

## 性能优化

### 调整参数
在 `step2_start_host_qwen3_14b_hetero.py` 中：
- `initial_launch_num`: 初始请求数量
- `avg_throughput`: 平均吞吐量
- `duration`: 运行时长

### 监控性能
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 查看结果
python3 step4_parse_results.py
```

## 注意事项

1. **层分配**: 当前配置为每GPU 10层，可根据显存调整
2. **网络延迟**: 跨主机通信延迟较大，影响性能
3. **模型加载**: 确保每个worker都能访问模型文件
4. **资源限制**: 2080Ti显存11GB，4090显存24GB，注意不要超限

## 文件清单

配置文件:
- `config/heterogeneous_2x2080ti_2x4090.ini` - 集群拓扑
- `config/machine_profile.ini` - 机器性能配置
- `layout/ilp_sol_qwen3_14b_2x2080ti_2x4090.ini` - 层分配方案
- `layout/simulator_cluster_2x2080ti_2x4090.ini` - 模拟器集群配置
- `config/real_sys_config_qwen3_14b_hetero.txt` - 生成的系统配置

脚本文件:
- `step1_generate_system_config_qwen3_14b_2x2080ti_2x4090.py` - 配置生成
- `step2_start_host_qwen3_14b_hetero.py` - Host启动
- `step3_start_worker_qwen3_14b_hetero.py` - Worker启动
- `step4_parse_results.py` - 结果解析
