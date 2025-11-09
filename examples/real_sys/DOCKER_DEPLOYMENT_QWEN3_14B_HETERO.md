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

## Docker Swarm 网络配置

### 重要说明
由于Helix的限制：
1. **每个GPU需要独立的IP地址**
2. **不支持端口映射**
3. **相邻节点必须能互相访问**

因此使用 **Docker Swarm 的 overlay 网络**来实现跨主机容器通信。

### 架构方案: Docker Swarm + Overlay Network

Docker Swarm的overlay网络允许不同主机上的容器在同一虚拟网络中通信，无需配置复杂的路由。

## 部署步骤

### 前置要求

1. 确保两台主机之间可以互相访问
2. 开放Docker Swarm所需端口：
   - TCP 2377 (集群管理)
   - TCP/UDP 7946 (节点通信)
   - UDP 4789 (overlay网络流量)

```bash
# 在两台主机上都执行
sudo ufw allow 2377/tcp
sudo ufw allow 7946/tcp
sudo ufw allow 7946/udp
sudo ufw allow 4789/udp
```

### 步骤1: 初始化Docker Swarm

#### 在主机1 (10.202.210.104) 上初始化Swarm管理节点
```bash
docker swarm init --advertise-addr 10.202.210.104
```

执行后会输出一个加入命令，类似：
```
docker swarm join --token SWMTKN-1-xxxxx 10.202.210.104:2377
```

#### 在主机2 (10.130.151.13) 上加入Swarm
```bash
# 使用上一步输出的命令
docker swarm join --token SWMTKN-1-xxxxx 10.202.210.104:2377
```

#### 验证集群状态（在主机1上）
```bash
docker node ls
```

应该看到两个节点都是 Ready 状态。

### 步骤2: 创建Overlay网络

在主机1上创建overlay网络：
```bash
docker network create \
  --driver overlay \
  --subnet=10.100.0.0/16 \
  --attachable \
  helix_overlay_network
```

注意：
- `--attachable` 允许普通容器连接到这个网络（不仅是服务）
- 使用独立的子网 10.100.0.0/16 避免与物理网络冲突

验证网络创建：
```bash
docker network ls | grep helix_overlay_network
```

### 步骤3: 在主机1启动Worker容器 (2080Ti)

```bash
# GPU1 容器 (10.100.0.11)
docker run -d \
  --name helix_worker_gpu1_2080ti \
  --network helix_overlay_network \
  --ip 10.100.0.11 \
  --gpus '"device=5"' \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.100.0.11"

# GPU2 容器 (10.100.0.12)
docker run -d \
  --name helix_worker_gpu2_2080ti \
  --network helix_overlay_network \
  --ip 10.100.0.12 \
  --gpus '"device=6"' \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.100.0.12"
```

### 步骤4: 在主机2启动Worker容器 (4090)

```bash
# GPU3 容器 (10.100.0.13)
docker run -d \
  --name helix_worker_gpu3_4090 \
  --network helix_overlay_network \
  --ip 10.100.0.13 \
  --gpus '"device=0"' \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.100.0.13"

# GPU4 容器 (10.100.0.14)
docker run -d \
  --name helix_worker_gpu4_4090 \
  --network helix_overlay_network \
  --ip 10.100.0.14 \
  --gpus '"device=1"' \

  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step3_start_worker_qwen3_14b_hetero.py maxflow 10.100.0.14"
```

### 步骤5: 测试跨主机网络连通性

```bash
# 从主机1的GPU1容器ping主机2的GPU3容器
docker exec helix_worker_gpu1_2080ti ping -c 3 10.100.0.13

# 从主机1的GPU1容器ping主机2的GPU4容器
docker exec helix_worker_gpu1_2080ti ping -c 3 10.100.0.14

# 从主机2的GPU3容器ping主机1的GPU1容器
docker exec helix_worker_gpu3_4090 ping -c 3 10.100.0.11
```

### IP地址分配方案

由于使用overlay网络，IP地址需要更新：

- **GPU1 (2080Ti)**: 10.100.0.11 - 层0-9
- **GPU2 (2080Ti)**: 10.100.0.12 - 层10-19
- **GPU3 (4090)**: 10.100.0.13 - 层20-29
- **GPU4 (4090)**: 10.100.0.14 - 层30-39
- **Coordinator**: 使用主机1的物理IP 10.202.210.104 或 10.100.0.1

## 完整部署流程

### 步骤0: 构建Docker镜像（两台主机都需要）

```bash
cd /root/Helix-ASPLOS25
docker build -t myhelix:latest -f- . <<EOF
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \\
    python3 python3-pip git wget curl \\
    && rm -rf /var/lib/apt/lists/*

# 安装conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \\
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:\$PATH

# 创建runtime环境
RUN conda create -n runtime python=3.10 -y

WORKDIR /Helix-ASPLOS25

COPY requirements.txt .
RUN /opt/conda/envs/runtime/bin/pip install -r requirements.txt

COPY . .
RUN /opt/conda/envs/runtime/bin/pip install -e .

CMD ["/bin/bash"]
EOF
```

### 步骤1: 生成系统配置（在主机1上执行）

**注意**：需要先修改配置文件使用overlay网络的IP地址。

编辑 `step1_generate_system_config_qwen3_14b_2x2080ti_2x4090.py`，将IP地址改为：
```python
# 原来的IP地址
# ip_addrs = ['10.202.210.105', '10.202.210.106', '10.130.151.14', '10.130.151.15']

# 改为overlay网络IP
ip_addrs = ['10.100.0.11', '10.100.0.12', '10.100.0.13', '10.100.0.14']
```

然后生成配置：
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python3 step1_generate_system_config_qwen3_14b_2x2080ti_2x4090.py
```

这将生成 `./config/real_sys_config_qwen3_14b_hetero.txt`

### 步骤2: 准备模型文件

确保模型文件在 `./model/` 目录下。两台主机都需要有相同的模型文件。

### 步骤3: 启动Coordinator（在主机1上执行）

Coordinator可以在宿主机运行，也可以在容器中运行。

**方案A - 在宿主机运行（推荐）**：
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
conda activate helix  # 或你的环境名
python3 step2_start_host_qwen3_14b_hetero.py offline maxflow
```

**方案B - 在容器中运行**：
```bash
docker run -it --rm \
  --network helix_overlay_network \
  --ip 10.100.0.1 \
  -v /root/Helix-ASPLOS25:/Helix-ASPLOS25 \
  myhelix:latest \
  bash -c "cd /Helix-ASPLOS25/examples/real_sys && /opt/conda/envs/runtime/bin/python3 step2_start_host_qwen3_14b_hetero.py offline maxflow"
```

### 步骤4: 启动Workers（按照上述步骤3和4）

在主机1和主机2上分别启动worker容器。

### 步骤5: 验证部署

```bash
# 检查各容器状态
docker ps

# 在主机1上查看
docker ps --filter name=helix_worker

# 在主机2上查看
docker ps --filter name=helix_worker

# 查看日志
docker logs helix_worker_gpu1_2080ti
docker logs helix_worker_gpu2_2080ti
docker logs helix_worker_gpu3_4090
docker logs helix_worker_gpu4_4090

# 测试overlay网络连通性
# 从GPU1容器测试到其他GPU
docker exec helix_worker_gpu1_2080ti ping -c 3 10.100.0.12  # GPU2
docker exec helix_worker_gpu1_2080ti ping -c 3 10.100.0.13  # GPU3
docker exec helix_worker_gpu1_2080ti ping -c 3 10.100.0.14  # GPU4

# 从GPU3容器测试回GPU1
docker exec helix_worker_gpu3_4090 ping -c 3 10.100.0.11  # GPU1
```

### 步骤6: 查看运行结果

```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python3 step4_parse_results.py
```



## Docker Swarm 管理命令

### 查看Swarm状态

```bash
# 查看节点
docker node ls

# 查看网络
docker network ls
docker network inspect helix_overlay_network

# 查看服务
docker service ls
```

### 离开Swarm集群

如果需要重新配置：

```bash
# 在worker节点上
docker swarm leave

# 在manager节点上
docker swarm leave --force
```

### 故障恢复

如果overlay网络出现问题：

```bash
# 在manager节点上删除并重建网络
docker network rm helix_overlay_network
docker network create \
  --driver overlay \
  --subnet=10.100.0.0/16 \
  --attachable \
  helix_overlay_network
```

## 网络架构说明

### Overlay网络工作原理

Docker Swarm的overlay网络使用VXLAN隧道技术：
- 容器之间的通信通过VXLAN封装
- 跨主机流量在UDP 4789端口传输
- 网络对容器透明，容器看到的是同一个虚拟网络

### IP地址规划

| 组件 | IP地址 | 物理位置 | GPU设备 | 推理层 |
|------|--------|---------|---------|--------|
| GPU1 | 10.100.0.11 | 主机1 | device=5 (2080Ti) | 0-9 |
| GPU2 | 10.100.0.12 | 主机1 | device=6 (2080Ti) | 10-19 |
| GPU3 | 10.100.0.13 | 主机2 | device=0 (4090) | 20-29 |
| GPU4 | 10.100.0.14 | 主机2 | device=1 (4090) | 30-39 |
| Coordinator | 10.100.0.1 (可选) | 主机1 | - | - |

### 流水线通信拓扑

```
Coordinator (10.100.0.1)
    ↓
GPU1 (10.100.0.11) → GPU2 (10.100.0.12) → GPU3 (10.100.0.13) → GPU4 (10.100.0.14)
    ↓                      ↓                      ↓                      ↓
返回给Coordinator (all-reduce或point-to-point)
```

## 防火墙配置

### 主机间防火墙规则

两台主机之间必须开放以下端口：

```bash
# 在主机1和主机2都执行
sudo ufw allow from 10.202.210.104
sudo ufw allow from 10.130.151.13

# 或者直接允许对方的整个网段
sudo ufw allow from 10.202.210.0/24
sudo ufw allow from 10.130.151.0/24

# Docker Swarm必需端口
sudo ufw allow 2377/tcp    # 集群管理通信
sudo ufw allow 7946/tcp    # 节点间通信
sudo ufw allow 7946/udp    # 节点间通信
sudo ufw allow 4789/udp    # overlay网络数据平面(VXLAN)

# 如果启用了防火墙，确保SSH可用
sudo ufw allow 22/tcp

# 重新加载防火墙
sudo ufw reload
sudo ufw status
```

### 测试物理网络连通性

在配置Swarm之前，先确保两台主机之间可以互相访问：

```bash
# 从主机1测试主机2
ping -c 3 10.130.151.13

# 从主机2测试主机1
ping -c 3 10.202.210.104
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
