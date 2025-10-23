# 单机4台RTX2080Ti运行LLaMA2-13B配置说明

## 修改内容

本配置已经修改为支持在单机上使用4台RTX2080Ti GPU运行LLaMA2-13B模型。

### 1. 硬件配置文件 (config/machine_profile.ini)
- 添加了 RTX2080Ti 的配置项
- VRAM大小: 11GB
- 网络带宽: 10Gbps

### 2. 集群配置文件 (config/single4.ini)
- 新建了4个GPU节点的集群配置
- 配置为全连接拓扑
- 所有节点类型为 RTX2080Ti

### 3. 模型配置文件 (model/config.json)
- 从 LLaMA2-70B 改为 LLaMA2-13B
- hidden_size: 8192 -> 5120
- intermediate_size: 28672 -> 13824
- num_attention_heads: 64 -> 40
- num_hidden_layers: 80 -> 40

### 4. 系统配置生成脚本 (step1_generate_system_config.py)
- host_ip 改为 "localhost"
- type2ips 改为使用4个 RTX2080Ti，所有IP为 "localhost"
- machine_num_dict 改为 {"RTX2080Ti": 4}
- model_name 改为 ModelName.LLaMa13B
- cluster文件改为 "./config/single4.ini"

### 5. 主机启动脚本 (step2_start_host.py)
- maxflow_offline 和 maxflow_online 函数更新配置
- machine_num_dict 改为 {"RTX2080Ti": 4}
- model_name 改为 ModelName.LLaMa13B
- cluster文件改为 "./config/single4.ini"

## 使用方法

### 前提条件
1. 确保系统有4张RTX2080Ti GPU
2. 安装所有必要的依赖包 (参考 requirements.txt)
3. 下载或准备 LLaMA2-13B 模型文件到 ./model/ 目录

### 运行步骤

#### 步骤1: 生成系统配置
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python step1_generate_system_config.py
```

#### 步骤2: 启动主机 (Host)
```bash
# 离线模式 + maxflow调度
python step2_start_host.py offline maxflow

# 或者在线模式 + maxflow调度
python step2_start_host.py online maxflow

# 或者使用其他调度方法 (swarm/random)
python step2_start_host.py offline swarm
python step2_start_host.py offline random
```

#### 步骤3: 启动工作节点 (Worker)
在另一个终端中运行:
```bash
python step3_start_worker.py maxflow
# 或者使用其他调度方法
python step3_start_worker.py swarm
python step3_start_worker.py random
```

#### 步骤4: 解析结果
```bash
python step4_parse_results.py
```

## 注意事项

1. **内存要求**: LLaMA2-13B模型在FP16精度下大约需要26GB显存，4张11GB的2080Ti可能需要使用模型并行或张量并行。

2. **网络配置**: 由于是单机配置，所有通信都在本地进行，网络延迟会非常低。

3. **GPU分配**: 确保每张GPU都能被正确识别，可以使用 `nvidia-smi` 检查。

4. **性能优化**: 
   - 可以调整 initial_launch_num 和 avg_throughput 参数来优化吞吐量
   - 根据实际性能调整 duration 参数

5. **模型文件**: 确保 ./model/ 目录包含正确的 LLaMA2-13B 模型文件:
   - config.json (已更新)
   - tokenizer.json
   - tokenizer_config.json
   - 模型权重文件

## 文件位置
- 配置文件: `./config/`
- 模型文件: `./model/`
- 布局方案: `./layout/`
- 运行结果: `./result/`
