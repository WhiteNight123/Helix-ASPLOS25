# ASPLOS'25 Helix
## 1. Introduction 
Helix is a distributed system designed for high-throughput, low-latency large language model
serving across heterogeneous and potentially geo-distributed GPU clusters. This repository
contains the official implementation of both Helix's simulator and prototype system. Our paper
can be found here [https://arxiv.org/abs/2406.01566](https://arxiv.org/abs/2406.01566).


## 2. Distributed LLM Serving Real System Tutorial
We build a prototype system for Helix using ZeroMQ as the communication framework and vLLM as the
execution engine. In the following example, we will install all dependencies on a fresh Ubuntu 24.04
LTS system and run Helix's prototype system to serve LLaMa 70B in a cluster with 24 machines.

### 2.1 Installing Dependencies
#### Basic C++ Building Tools
```bash
sudo apt update
sudo apt install build-essential
sudo apt install cmake
```
After this step, run the following commands to verify installation:
```bash
gcc --version   # gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0
cmake --version # cmake version 3.28.3
```

#### CUDA and GPU Driver
We install `CUDA 12.6` following [NVIDIA's official documentation](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network).
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-6
```
Then, we set the environment variables and reboot the system to complete installation:
```bash
echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
sudo reboot
```
To verify installation, run:
```bash
nvcc --version        # Cuda compilation tools, release 12.6, V12.6.77
nvidia-smi --version  # DRIVER version      : 560.35.03
```

#### ZeroMQ
We use ZeroMQ as the communication framework. To install `libzmq` and its C++ binding `cppzmq`,
follow the steps in [this GitHub repo](https://github.com/zeromq/cppzmq).

#### Pybind11
We implement the inter-node communication logic in C++. In order to call the C++ functions from Python
side, we use `pybind11`:
```bash
sudo apt-get install pybind11-dev
```

#### Python Dependencies
First, we set python as python3 and install `pip`:
```bash
sudo apt install python-is-python3
sudo apt install python3-pip
```
Then, we install `conda` to isolate the Python environment we are using:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
Remember to initialize conda in the shell environment before using it. We create a new conda environment
with:
```bash
conda create -n runtime python=3.10 -y && conda activate runtime
```
We use `vllm` as our execution engine. For the prototype system, we require using `vllm 0.4.0.post1`.
We can install this version of `vllm` and `numpy` using:
```bash
pip install vllm==0.4.0.post1
pip install numpy~=1.26
```

#### Runtime within Conda Environment
Run the following command to install `libstdcxx` in the conda environment, this can avoid errors like
`'GLIBCXX_3.4.32' not found` in later steps:
```bash
conda install -c conda-forge libstdcxx-ng
```

### 2.2 Installing Helix's Communication Framework
First, at the root directory of this repository, run the following command to install the directory:
```bash
pip install -e .
```

Then, at the root directory of this repository, execute the following commands:
```bash
cd llm_sys/comm
bash build.sh
```
The build script will automatically build and install Helix's communication framework.

> **Tips:** You may need to change `CMAKE_PREFIX_PATH` if your conda environment has a different path
> from the default one. Also, you may need to change the file path in `setup.py` if the `.so` files
> you built have a different name from the one list there.

> **Tips:** By default, Helix's communication framework use ports starting from 6000 for inter-node
> communication. If you want to use other ports, you can change the `BASE_PORT` in `src/const.h`

To verify Helix's communication framework is correctly installed, we provide several unit tests.

#### Message Encoding & Decoding
```bash
cd build
./test_msg  # a unit test for message encoding & decoding
```
You should be able to see `Test Passed!` after some other logs.

#### Cross-Node Communication
On two machines, run the following command (replace the IP and port):
```bash
./packed_server 10.128.0.13 5555 1  # on machine 1
./packed_server 10.128.0.14 5555 1  # on machine 2
```
On one client machine, run the following command (replace the IP and port):
```bash
./packed_client tcp://10.128.0.13:5555 tcp://10.128.0.14:5555
```
If running correctly, the client will print out messages like this from both servers:
```
Received:
Creation time: 1730430003755545
Latency: 562 us
From server: 1
```

#### Python Binding Test
Run the following command in Python: 
```
import llm_host, llm_worker
```
If everything is correct, you should not receive any error messages.

### 2.3 Running Helix's Prototype System
With all dependencies and the communication framework installed, we can now start running Helix's
prototype system. Starting from the root directory of this repository, enter the example directory:
```bash
cd examples/real_sys
```
Here, we assume that you have already followed the simulation steps to generate the cluster
configuration files (`config/single24.ini` and `config/machine_profile.ini`) and the model
placement files (`layout/ilp_sol.ini` and `simulator_cluster.ini`). If you have not yet done so,
you can refer to Step 1 and Step 2 in the simulator tutorial. 

We also assume that the model to serve is stored in `model`. For the prototype system, we use
dummy weights. Therefore, you only need to provide the model config (`model/config.json`) and
tokenizer (`tokenizer.json` and `tokenizer_config.json`). We follow the standard format used
on [HuggingFace](https://huggingface.co/).

> **Tips:** Before running the following commands, make sure you are using the conda
> environment we just created. You can activate the environment using `conda activate runtime`.

Based on the files above, we can generate the system configuration file for Helix's runtime
system:
```bash
python step1_generate_system_config.py
```
Running this script generates `config/real_sys_config.txt`, which specifies the layers each
machine should hold and the connection to setup between machines. 

> **Tips:** To run on your own cluster, you need to change the IP addresses in 
> `step1_generate_system_config.py`. You also need to change the `CONFIG_BROADCAST_ADDR`
> in `llm_sys/utils.py` to the broadcast address of your cluster. (i.e. the IP address
> of the host machine).

After this step, we can deploy Helix to serve the model using the cluster. Before starting
the deployment, make sure you have a copy of Helix on every machine you are going to use.

In this example, we will deploy Helix using its MaxFlow-based scheduling method and run in
online mode, where the request arrival rate follows the distribution generated from Azure
Conversation Dataset. (Please refer to our paper for more details about online and offline
setup). 

On the host machine, run the following command:
```bash
python step2_start_host.py online maxflow
```
Then, on the worker machines, run the following command:
```bash
python step3_start_worker.py maxflow
```

> **Tips:** We design the prototype system to make it agnostic to the order of starting the
> host and worker machines. You can start the workers first and then the host, or vice versa.
> However, you need to make sure the scheduling method of the host matches that of the worker.

We also provide examples for other scheduling methods and setups:
```bash
# maxflow + offline
python step2_start_host.py offline maxflow  # on host
python step3_start_worker.py maxflow        # on workers

```

After running the above commands, you should see the host machine store two log files in the
`result` directory. The `events.txt` stores the launch and finish time of each iteration for
each query. To analyze this file, run:
```bash
python step4_parse_results.py
```
The `query_route.txt` stores the route each request takes. The format is like the following:
```bash
(0, 194, 96, [16, 15, 1, 17, 18, 3, 21, 8, 9, 2, 14, 20, 19, 6, 11, 7, 5, 0], [0, 3, 7, 11, 14, 23, 25, 26, 30, 34, 38, 42, 46, 53, 60, 68, 77, -1], [3, 7, 11, 14, 23, 25, 26, 30, 34, 38, 42, 46, 53, 60, 68, 77, 80, -1])
```
Here, the first three numbers are the query id, the input length and the output length. The
three arrays are the compute nodes used, start layer ids, and end layer ids.