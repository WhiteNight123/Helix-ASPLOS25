# Qwen3-32B-AWQ Setup on 2x RTX2080Ti + 1x RTX4090 GPUs

This guide provides instructions for running Qwen3-32B-AWQ model on a heterogeneous GPU setup with 2x RTX2080Ti and 1x RTX4090.

## Hardware Configuration

- **2x RTX2080Ti GPUs**: 11GB VRAM each (IP: 127.0.0.1, 127.0.0.2)
- **1x RTX4090 GPU**: 24GB VRAM (IP: 10.130.151.21)

## Layer Distribution

The Qwen3-32B model has 64 layers distributed as follows:
- **RTX2080Ti #1 (127.0.0.1)**: Layers 0-15 (16 layers)
- **RTX2080Ti #2 (127.0.0.2)**: Layers 16-31 (16 layers)
- **RTX4090 (10.130.151.21)**: Layers 32-63 (32 layers)

The RTX4090 handles more layers due to its larger VRAM capacity.

## Prerequisites

1. **Model Weights**: Ensure you have the Qwen3-32B-AWQ (4-bit quantized) model weights in the `./model` directory.
2. **Network**: All GPUs should be accessible via the network.
3. **Python Environment**: Helix environment with all dependencies installed.

## Setup Steps

### Step 1: Generate System Configuration

Run the configuration generator:

```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python3 step1_generate_system_config_qwen32b_3gpu_mixed_awq.py
```

This will create:
- `./config/real_sys_config_qwen32b_3gpu_mixed_awq.txt`: System configuration file
- Layout files in `./layout/`

### Step 2: Start the Host

Start the coordinator/host process with your chosen scheduling method:

```bash
# For maxflow scheduler in offline mode
python3 step2_start_host_qwen32b_3gpu_mixed_awq.py offline maxflow

# For maxflow scheduler in online mode
python3 step2_start_host_qwen32b_3gpu_mixed_awq.py online maxflow

# For swarm scheduler in offline mode
python3 step2_start_host_qwen32b_3gpu_mixed_awq.py offline swarm

# For random scheduler in offline mode
python3 step2_start_host_qwen32b_3gpu_mixed_awq.py offline random
```

The host must be started on the machine at **10.130.151.21**.

### Step 3: Start Workers

Start a worker process on each GPU. Open a separate terminal for each worker.

#### On RTX2080Ti #1 (127.0.0.1):
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python3 step3_start_worker_qwen32b_3gpu_mixed_awq.py maxflow 127.0.0.1
```

#### On RTX2080Ti #2 (127.0.0.2):
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python3 step3_start_worker_qwen32b_3gpu_mixed_awq.py maxflow 127.0.0.2
```

#### On RTX4090 (10.130.151.21):
```bash
cd /root/Helix-ASPLOS25/examples/real_sys
python3 step3_start_worker_qwen32b_3gpu_mixed_awq.py maxflow 10.130.151.21
```

**Note**: Replace `maxflow` with your chosen scheduling method (swarm, random) to match Step 2.

### Step 4: Parse Results

After the experiment completes, parse the results:

```bash
python3 step4_parse_results.py ./result/qwen32b_3gpu_mixed_awq_maxflow_offline/
```

Adjust the result directory path based on your scheduling method and mode.

## Configuration Files

- `./config/single3_mixed_awq.ini`: Cluster topology configuration
- `./config/machine_profile.ini`: GPU specifications (includes RTX4090 profile)
- `./layout/ilp_sol_qwen32b_3gpu_mixed_awq.ini`: Layer placement solution
- `./layout/simulator_cluster_qwen32b_3gpu_mixed_awq.ini`: Simulator cluster configuration

## Scheduling Methods

- **maxflow**: Max-flow based scheduling (recommended for best performance)
- **swarm**: Swarm-based heuristic scheduling
- **random**: Random scheduling (baseline)

## Modes

- **offline**: Batch processing mode with initial request launch
- **online**: Continuous request arrival mode with specified throughput

## Troubleshooting

1. **Connection Issues**: Verify that all IPs are reachable from the host machine.
2. **VRAM Errors**: Ensure AWQ quantized model weights are used, not full precision.
3. **Layer Mismatch**: Verify the layer distribution in the configuration files matches your setup.
4. **Port Conflicts**: Ensure no other processes are using the required ports.

## Performance Tips

1. Use `maxflow` scheduler for optimal performance
2. Monitor GPU utilization on each device
3. Adjust `vram_usage` parameter in worker scripts if needed (default: 0.9)
4. For better throughput, tune the `avg_throughput` parameter in online mode

## Results

Results will be saved in `./result/qwen32b_3gpu_mixed_awq_<scheduler>_<mode>/` with:
- Request latency statistics
- Throughput measurements
- GPU utilization metrics
- Scheduling decisions log
