# llm sys must be imported before torch
# use this version of vllm for execution engine
# pip install git+https://github.com/vllm-project/vllm.git@v0.4.0.post1
import time
import llm_worker
import torch
import os

from vllm.config import SchedulerConfig, CacheConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import Sequence, SequenceData, SequenceGroup, SequenceGroupMetadata, SequenceStatus
from vllm import SamplingParams

from llm_sys.engine.common import PipelineSequence, PipelineSequenceData, PipelineStageOut
from llm_sys.engine.scheduler import LayerwiseScheduler
from llm_sys.engine.exec_engine import PipelineStageEngine
# import llm_sys.engine.llama
# import llm_sys.engine.qwen2
import llm_sys.engine.qwen3

import llm_sys.utils as utils


def init_engine(layer_ids, model_name, vram_usage=0.8, quantization=None):
    """
    Initialize the pipeline stage engine.
    
    Args:
        layer_ids: List of layer IDs to run on this worker
        model_name: Path to model directory
        vram_usage: GPU memory utilization (0.0-1.0)
        quantization: Quantization method (None, "awq", "gptq", etc.)
    """
    engine_args = EngineArgs(model=model_name, block_size=16,
                             load_format="dummy", enforce_eager=True,
                             swap_space=8, max_num_batched_tokens=2048,
                             gpu_memory_utilization=vram_usage, dtype="float16",
                             quantization=quantization, max_num_seqs=256,
                             max_model_len=2048)

    engine = PipelineStageEngine.from_engine_args(engine_args, layer_ids)
    return engine


def run_and_submit(engine, start_idx, end_idx, is_last_layer, hidden_size, force_decode, worker_id, batch_size, num_prefill, num_decode, req_ids) -> bool:
    """
    :return: whether parsed prompt in this iter
    """
    # Step 2.2: run inference layer by layer
    finished_seq_infos, output = [], None

    # **************************intrument*************************
    torch.cuda.synchronize()
    cur_stage = ''
    if force_decode:
        cur_stage = 'decode'
    else:
        cur_stage = 'prefill'
    if batch_size > 0:
        with open(f'/Helix-ASPLOS25/examples/real_sys/log/measurement_{worker_id}.log', 'a') as f:
            print(f'{0} {batch_size} compute starts ({num_prefill}P+{num_decode}D) at {time.time()}', file = f)
    # **************************intrument*************************

    for layer_id in range(start_idx, end_idx):
        output = engine.step(layer_id=layer_id, finished_seq_infos=finished_seq_infos,
                             force_decode=force_decode)


    # **************************intrument*************************
    torch.cuda.synchronize()
    if batch_size > 0:
        with open(f'/Helix-ASPLOS25/examples/real_sys/log/measurement_{worker_id}.log', 'a') as f:
            print(f'{0} {batch_size} compute ends ({num_prefill}P+{num_decode}D) at {time.time()}', file = f)
    # **************************intrument*************************

    # Step 2.3: Prepare output
    if output == (None, None, None):
        # nothing to schedule, no need to re-enter
        return False
    if not is_last_layer:
        # output infos: list[((req_id (str), layer_id (int)), begin_idx (int), end_idx (int))]
        # output_tensor: a large tensor
        output_info_list, output_tensor, parsed_prompt = output
        output_tensor = output_tensor.cpu()

        # build finished reqeust information
        finished_ids, finished_offsets, finished_lengths = [], [], []
        # non-last, send out activations
        for output_info in output_info_list:
            finished_ids.append(int(output_info[0][0]))
            finished_offsets.append(output_info[1] * hidden_size)
            finished_lengths.append((output_info[2] - output_info[1]) * hidden_size)
    else:
        # last, send out int tensors (generated token ids)
        finished_ids, finished_offsets, finished_lengths = [], [], []
        generated_tokens = []
        parsed_prompt = output[1]
        for request_output in output[0]:
            cur_request_id = int(request_output.request_id[0])
            finished_ids.append(cur_request_id)
            finished_offsets.append(len(generated_tokens))
            finished_lengths.append(1)
            generated_tokens.append(1)
        output_tensor = torch.tensor(generated_tokens, dtype=torch.int32)

    # ------------------------------------------------------------------------------------------- #
    # Step 3: submit results
    # start_idx_list: List[int], offset in number of elements
    # length_list: List[int], length in number of elements
    # results_tensor: one large CPU tensor

    # **************************intrument*************************
    torch.cuda.synchronize()
    if batch_size > 0:
        with open(f'/Helix-ASPLOS25/examples/real_sys/log/measurement_{worker_id}.log', 'a') as f:
            print(f'{0} {req_ids} trans starts at {time.time()}', file = f)
    # **************************intrument*************************

    llm_worker.submit_requests(finished_ids, finished_offsets, finished_lengths, output_tensor)
    
    # ------------------------------------------------------------------------------------------- #
    return parsed_prompt


def run_worker(scheduling_method: str, model_name: str, worker_ip: str = None, vram_usage=0.8, quantization=None):
    # warm up gpu and initialize llm_sys
    print("[Python] Starting worker initialization...")
    utils.warm_up()
    print("[Python] GPU warm-up completed.")
    
    if worker_ip is None:
        worker_ip: str = utils.get_local_ip()
        assert worker_ip.startswith("10"), "Local IP must start with 10"
    
    # Remove old log file if exists
    worker_id = worker_ip.split('.')[-1]
    log_file = f'server{worker_id}.log'
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"[Python] Removed old log file: {log_file}")
    
    print(f"[Python] Starting network threads with IP: {worker_ip}")
    llm_worker.start_network_threads(utils.CONFIG_BROADCAST_ADDR, worker_ip, scheduling_method)
    print("[Python] Network threads started, getting model layer indices...")
    
    start_idx, end_idx, is_last_layer = llm_worker.get_model_start_end_idx()
    print(f"[Python] Cluster initialization finished!")
    print(f"[Python] Model layers: [{start_idx}, {end_idx}).")
    print(f"[Python] Does this node output the last layer: {is_last_layer}.")

    # init vllm
    quantization_str = f" with {quantization} quantization" if quantization else ""
    print(f"[Python] Initializing vLLM engine for layers {start_idx} to {end_idx}{quantization_str}...")
    layer_ids = list(range(start_idx, end_idx))
    engine: PipelineStageEngine = init_engine(layer_ids, model_name, vram_usage=vram_usage, quantization=quantization)
    # Store worker_id in model_config for logging purposes
    engine.model_config.worker_id = worker_id
    hidden_size = engine.model_config.get_hidden_size()
    print(f"[Python] vLLM engine initialized successfully! Hidden size: {hidden_size}")
    
    # Calculate and print memory usage using vLLM's profiler and cache engine
    # Get total GPU memory
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
    
    # Get model weights memory from vLLM's CudaMemoryProfiler (accurate measurement)
    # This is measured during model loading in model_runner.load_model()
    model_memory_usage = engine.model_executor.driver_worker.model_runner.model_memory_usage / (1024 ** 3)  # Convert to GB
    
    # Get KV cache size using vLLM's interfaces
    num_gpu_blocks = engine.cache_config.num_gpu_blocks
    block_size = engine.cache_config.block_size
    cache_dtype = engine.cache_config.cache_dtype
    
    # Method 1: Use vLLM's get_cache_block_size_bytes interface
    # This gives us the exact size per block as calculated by vLLM
    bytes_per_block_vllm = engine.model_executor.driver_worker.get_cache_block_size_bytes(
        block_size, cache_dtype
    )
    total_kv_cache_size_vllm = (num_gpu_blocks * bytes_per_block_vllm) / (1024 ** 3)  # Convert to GB
    
    # Method 2: Get actual GPU cache tensor size
    # The gpu_cache is a list of [key_cache, value_cache] tensors
    gpu_cache = engine.model_executor.driver_worker.gpu_cache
    if gpu_cache is not None and len(gpu_cache) >= 2:
        # gpu_cache[0] is key_cache, gpu_cache[1] is value_cache
        key_cache_size = gpu_cache[0].element_size() * gpu_cache[0].nelement() / (1024 ** 3)
        value_cache_size = gpu_cache[1].element_size() * gpu_cache[1].nelement() / (1024 ** 3)
        total_kv_cache_size_actual = key_cache_size + value_cache_size
    else:
        total_kv_cache_size_actual = 0
    
    # Get current GPU memory usage (including everything - model, KV cache, and other allocations)
    current_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
    current_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # Convert to GB
    
    print(f"[Python] ========== Memory Usage (vLLM Profiler & Cache Engine) ==========")
    print(f"[Python] Total GPU Memory: {total_gpu_memory:.2f} GB")
    print(f"[Python] Model Weights (Profiled): {model_memory_usage:.2f} GB")
    print(f"[Python] KV Cache (vLLM method): {total_kv_cache_size_vllm:.2f} GB ({num_gpu_blocks} blocks)")
    print(f"[Python] KV Cache (Actual tensor): {total_kv_cache_size_actual:.2f} GB")
    print(f"[Python]   - Block size: {block_size} tokens, {bytes_per_block_vllm / (1024**2):.4f} MB per block")
    print(f"[Python] Current GPU Allocated: {current_allocated:.2f} GB")
    print(f"[Python] Current GPU Reserved: {current_reserved:.2f} GB")
    print(f"[Python] Available for other processes: {total_gpu_memory - current_reserved:.2f} GB")
    print(f"[Python] =====================================================================")
    print(f"[Python] Entering main inference loop...")

    # Extract worker ID from IP address (e.g., 10.100.0.13 -> worker_13)
    worker_id = worker_ip.split('.')[-1]
    
    # Use absolute path for log file to ensure it's written to the correct location
    log_dir = '/Helix-ASPLOS25/examples/real_sys/log'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'worker_{worker_id}.log')
    
    # Remove old log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        print(f"[Python] Removed old log file: {log_file_path}")
    
    print(f"[Python] Log file will be written to: {log_file_path}")
    
    # Remove old measurement log file if it exists
    measurement_log_path = os.path.join(log_dir, f'measurement_{worker_id}.log')
    if os.path.exists(measurement_log_path):
        os.remove(measurement_log_path)
        print(f"[Python] Removed old measurement log file: {measurement_log_path}")
    
    last_log_time = time.time()

    while True:
        # ------------------------------------------------------------------------------------------- #
        # Step 1: fetch new requests to compute on
        # request_ids: List[int]
        # is_prompt_list: List[bool]
        # start_layer_idx_list: List[int]
        # end_layer_idx_list: List[int]
        # num_tokens_list: List[int]
        #   1. for prompt, its the number of input tokens
        #   2. for decode, its the context size
        # max_tokens_list: List[int]
        #   1. in decode phase, if num_tokens (context) = max_tokens - 1, then we can release kv cache
        #   2. FIXME: be careful about the case when decode has length 0
        # offsets: List[int] (in number of elements)
        # lengths: List[int] (in number of elements)
        # is_token_tensor_list: List[bool]
        #   1. if true, offset and length belong to token tensor
        #   2. if false, offset and length belong to activation tensor
        # token_tensor: Torch.tensor on CUDA:0, torch.int32 (token ids)
        # activation_tensor: Torch.tensor on CUDA:0, torch.fp16 (activations)
        request_ids, is_prompt_list, start_layer_idx_list, end_layer_idx_list, num_tokens_list, max_tokens_list, \
            offsets, lengths, is_token_tensor_list, token_tensor, activation_tensor = llm_worker.fetch_new_requests()


        # **************************intrument*************************
        batch_size = len(request_ids)
        if batch_size > 0:
            with open(f'/Helix-ASPLOS25/examples/real_sys/log/measurement_{worker_id}.log', 'a') as f:
                cur = time.time()
                for rid in request_ids:
                    print(f'request {rid} arrives at worker at {cur}', file = f)
                print(f'{0} {request_ids} recv at {cur}', file=f)
        num_prefill = 0
        num_decode = 0
        # **************************intrument*************************


        # ------------------------------------------------------------------------------------------- #
        # Step 2: run vllm
        # Step 2.1: put requests into the queues (register in vllm)
        for (request_id, is_prompt, start_layer_idx, end_layer_idx, num_tokens, max_tokens, offset,
             length, is_token) in zip(
            request_ids, is_prompt_list, start_layer_idx_list, end_layer_idx_list, num_tokens_list,
            max_tokens_list, offsets, lengths, is_token_tensor_list
        ):
            if is_prompt:

                # **************************intrument*************************
                num_prefill += 1
                # **************************intrument*************************

                print(f"[Prompt] Request {request_id} arrives (input_len={num_tokens}, max_len={max_tokens}, "
                      f"layers=[{start_layer_idx}, {end_layer_idx}))")
                # prompt phase requests
                if is_token:
                    # first layer: input tensor should be none
                    assert start_layer_idx == 0, "Only request that will infer layer 0 use token tensor!"
                    prompt_token_ids = list(token_tensor[offset: offset + length])
                    sampling_params = SamplingParams()
                    sampling_params.max_tokens = max_tokens - num_tokens  # it's generated tokens
                    sampling_params.ignore_eos = True
                    engine.add_request(f"{request_id}", "", sampling_params,
                                       local_layers=(start_layer_idx, end_layer_idx - 1),  # inclusive, thus -1
                                       seq_id=request_id, input_tensor=None,
                                       prompt_token_ids=prompt_token_ids)
                else:
                    # non-first layer: prompt token ids should be none
                    assert not start_layer_idx == 0, "Request that will infer layer 0 must use token tensor!"
                    input_tensor = activation_tensor[offset: offset + length].reshape(num_tokens, hidden_size)
                    sampling_params = SamplingParams()
                    sampling_params.max_tokens = max_tokens - num_tokens  # it's generated tokens
                    sampling_params.ignore_eos = True
                    engine.add_request(f"{request_id}", "", sampling_params,
                                       local_layers=(start_layer_idx, end_layer_idx - 1),  # inclusive, thus -1
                                       seq_id=request_id, input_tensor=input_tensor,
                                       prompt_token_ids=[1] * num_tokens)
            else:

                # **************************intrument*************************
                num_decode += 1
                # **************************intrument*************************

                # print(f"[Decode] Request {request_id} arrives (context_len={num_tokens}, max_len={max_tokens}, "
                #       f"layers=[{start_layer_idx}, {end_layer_idx}))")
                if is_token:
                    # first layer: no activations
                    engine.scheduler.update_req_data(start_layer_idx,
                                                     (f"{request_id}", start_layer_idx),
                                                     {(request_id, start_layer_idx): None})
                else:
                    input_tensor = activation_tensor[offset: offset + length].reshape(1, hidden_size)
                    engine.scheduler.update_req_data(start_layer_idx,
                                                     (f"{request_id}", start_layer_idx),
                                                     {(request_id, start_layer_idx): input_tensor})

        # step 2.2 & 2.3: run vllm and submit
        parsed_prompt = run_and_submit(engine=engine, start_idx=start_idx, end_idx=end_idx, is_last_layer=is_last_layer,
                                       hidden_size=hidden_size, force_decode=False, worker_id=worker_id, batch_size=batch_size, num_prefill=num_prefill, num_decode=num_decode, req_ids=request_ids)
        if parsed_prompt:
            parsed_prompt = run_and_submit(engine=engine, start_idx=start_idx, end_idx=end_idx,
                                           is_last_layer=is_last_layer, hidden_size=hidden_size,
                                           force_decode=True, worker_id=worker_id, batch_size = batch_size, num_prefill=num_prefill, num_decode=num_decode, req_ids=request_ids)
            assert not parsed_prompt, "Parsed prompt twice!"

        num_total_gpu = engine.cache_config.num_gpu_blocks
        num_free_gpu = engine.scheduler.block_manager.get_num_free_gpu_blocks()
        gpu_cache_usage = 1.0 - (num_free_gpu / num_total_gpu)

        
        # log kv cache status
        if time.time() > last_log_time + 2:
            with open(log_file_path, 'a') as f:
                print(f'memory stat log time: {time.time()}, num_free_gpu: {num_free_gpu}, num_total_gpu: {num_total_gpu}, gpu_cache_usage_sys: {gpu_cache_usage * 100}%', file=f)

            last_log_time = time.time()
            # GPU KV Cache Usage in %.
            num_total_gpu = engine.cache_config.num_gpu_blocks
            num_free_gpu = engine.scheduler.block_manager.get_num_free_gpu_blocks()
            gpu_cache_usage = 1.0 - (num_free_gpu / num_total_gpu)

            # CPU KV Cache Usage in %
            num_total_cpu = engine.cache_config.num_cpu_blocks
            cpu_cache_usage = 0.
            if num_total_cpu > 0:
                num_free_cpu = engine.scheduler.block_manager.get_num_free_cpu_blocks()
                cpu_cache_usage = 1.0 - (num_free_cpu / num_total_cpu)

            print(f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}% - "
                  f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%.")
            if cpu_cache_usage > 0.1:
                print("Warning: CPU KV cache usage is larger than 0.1, considering lower arrival rate!")
