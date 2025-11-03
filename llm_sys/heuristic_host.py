# 2024.04.25 Yixuan Mei

import llm_host
import os
import time
import random

from simulator.trace_generator.sharegpt_loader import ShareGPTLoader
from llm_sys.utils import get_local_ip, CONFIG_BROADCAST_ADDR, FlyingQuery


def run_heuristic_host_online(
        # scheduler
        scheduler_name: str,
        # cluster
        real_sys_config_file_name: str,
        # throughput
        avg_throughput: float,
        duration: int,
        # result
        result_logging_dir: str,
) -> None:
    """
    Run host with !!![Swarm/Random + Online mode]!!!.
    """
    assert scheduler_name == "swarm" or scheduler_name == "random", "Scheduler must be either swarm or random!"
    print(f"Initializing host with {scheduler_name} scheduling!")

    # ------------------------------------- Online Generator (Poisson arrival with ShareGPT) ------------------------------------ #
    # Load ShareGPT dataset
    loader = ShareGPTLoader("/mnt/lvm-data/home/dataset/sharegpt/common_en_70k.jsonl")
    print('Loading ShareGPT dataset into memory...')
    loader.load_data()
    print('ShareGPT dataset loaded successfully!')
    
    # Generate Poisson arrival trace with ShareGPT prompts
    # avg_throughput is tokens/second, we need to convert to requests/second
    # Assuming average tokens per request ~ 512 (256 input + 256 output)
    avg_rps = avg_throughput / 512.0
    trace = []  # list of (time, input_length, output_length)
    current_time = 0.0
    random.seed(0)
    
    while current_time < duration:
        # Get a random prompt from ShareGPT
        qa_pair = loader.get_random_qa()
        prompt = qa_pair.get('human', '')
        
        # Estimate input length from prompt (approximate: words * 1.3 for tokens)
        # Limit to reasonable range to avoid KV cache overflow
        estimated_tokens = int(len(prompt.split()) * 1.3)
        input_length = max(32, min(estimated_tokens, 256))  # Clamp between 32 and 256
        # Random output length between 50 and 256 tokens
        output_length = random.randint(50, 256)
        
        trace.append((current_time, input_length, output_length))
        
        # Generate next arrival time using exponential distribution (Poisson process)
        inter_arrival_time = random.expovariate(avg_rps)
        current_time += inter_arrival_time
    
    print(f"Generated {len(trace)} requests with Poisson arrival for {duration}s duration")
    # ------------------------------------------------------------------------------------------- #

    # ------------------------------------- Init System ------------------------------------ #
    host_ip: str = get_local_ip()
    assert host_ip.startswith("10"), "Local IP must be of form 10.xxx.xxx.xxx"
    llm_host.start_network_threads(CONFIG_BROADCAST_ADDR, host_ip, real_sys_config_file_name, scheduler_name)
    time.sleep(20)
    print("[Python] Cluster initialization finished!")
    # -------------------------------------------------------------------------------------- #
    ground_zero = time.time()
    next_query_id = 0
    flying_queries_dict = {}
    # -----  log items ----- #
    # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
    query_routes = []
    # time - query id - in/out - phase - context_len - this_iter_processed
    events = []
    # ---------------------- #
    while True:
        # get time
        now = time.time() - ground_zero
        if now > duration + 30:
            break

        # send new requests into cluster if needed
        while not len(trace) == 0 and trace[0][0] <= now:
            # the request has a time stamp smaller than now, should be sent
            expected_submit_time, input_length, output_length = trace.pop(0)

            # get query id
            cur_query_id = next_query_id
            next_query_id += 1

            # send it into the cluster (system will take care of routing)
            llm_host.launch_request(
                "prompt",  # request_type
                cur_query_id,  # request_id
                input_length,  # num_tokens
                input_length + output_length,  # max_num_tokens
                [i for i in range(input_length)],  # token_ids
                False,  # set_routing
                [],  # server_ids
                [],  # start_layer_ids
                [],  # end_layer_ids
            )

            # put into flying queries
            flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                            input_length=input_length,
                                                            output_length=output_length,
                                                            compute_node_uids=None,
                                                            start_layers=None,
                                                            end_layers=None,
                                                            pipeline=None)

            # save log
            # routing info will be available when we receive the request from cluster
            # time - query id - in/out - phase - context_len - this_iter_processed
            events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
            print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                  f"max_len = {input_length + output_length}")

        # get finished requests
        now = time.time() - ground_zero
        finished_query_ids, generated_token_ids, routes, num_layers = llm_host.gather_finished_requests()
        for query_uid, route_list, num_layer_list in zip(finished_query_ids, routes, num_layers):
            # first receive the message
            py_on_the_fly_query = flying_queries_dict[query_uid]
            if py_on_the_fly_query.processed_tokens == 0:
                # prompt phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1))
                py_on_the_fly_query.processed_tokens += py_on_the_fly_query.input_length + 1

                # now we can log the request with its route
                # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
                cur_log_start = 0
                start_layer_ids, end_layer_ids = [], []
                for num_layer in num_layer_list:
                    start_layer_ids.append(cur_log_start)
                    end_layer_ids.append(cur_log_start + num_layer)
                    cur_log_start += num_layer
                query_routes.append((query_uid, py_on_the_fly_query.input_length, py_on_the_fly_query.output_length,
                                     route_list, start_layer_ids, end_layer_ids))
            else:
                # decode phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1))
                py_on_the_fly_query.processed_tokens += 1

            # then we decide whether to send out new messages (decodes)
            max_size = py_on_the_fly_query.input_length + py_on_the_fly_query.output_length
            assert py_on_the_fly_query.processed_tokens <= max_size, "Found request that did not end!"
            if py_on_the_fly_query.processed_tokens == max_size:
                # not send: finished, remove from expectations
                del flying_queries_dict[query_uid]
                print(f"Query {query_uid}, finished (total_len={py_on_the_fly_query.processed_tokens})")

            else:
                # then we send the query back into the cluster
                llm_host.launch_request(
                    "decode",  # request_type
                    query_uid,  # request_id
                    py_on_the_fly_query.processed_tokens,  # num_tokens (context size)
                    max_size,  # max_num_tokens
                    [-1],  # token_ids
                    False,  # set_routing
                    [],  # server_ids
                    [],  # start_layer_ids
                    [],  # end_layer_ids
                )

                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1))

    # save logging files
    print(f"Queries still flying: {flying_queries_dict.keys()}.")
    query_routes_file_name = os.path.join(result_logging_dir, "query_route.txt")
    events_file_name = os.path.join(result_logging_dir, "events.txt")
    with open(query_routes_file_name, "w") as f:
        for item in query_routes:
            f.write(f"{item}\n")
    with open(events_file_name, "w") as f:
        for item in events:
            f.write(f"{item}\n")


def run_heuristic_host_offline(
        # scheduler
        scheduler_name: str,
        # cluster
        real_sys_config_file_name: str,
        # throughput
        initial_launch_num: int,
        duration: int,
        # result
        result_logging_dir: str,
) -> None:
    """
    Run host with !!![Swarm/Random + Offline mode]!!!.
    """
    assert scheduler_name == "swarm" or scheduler_name == "random", "Scheduler must be either swarm or random!"
    print(f"Initializing host with {scheduler_name} scheduling!")

    # ------------------------------------- Offline Initial (Burst with ShareGPT) ------------------------------------- #
    # Load ShareGPT dataset
    loader = ShareGPTLoader("/mnt/lvm-data/home/dataset/sharegpt/common_en_70k.jsonl")
    print('Loading ShareGPT dataset into memory...')
    loader.load_data()
    print('ShareGPT dataset loaded successfully!')
    
    # Generate burst-style initial requests uniformly distributed in a short time window
    initial_requests = []
    window_seconds = 1.0  # Concentrate all initial requests in 1 second
    random.seed(0)
    
    # Generate uniformly distributed time stamps within the window
    time_stamps = []
    interval = window_seconds / initial_launch_num
    for i in range(initial_launch_num):
        base_time = 0.1 + i * interval
        # Add small jitter
        jitter = random.uniform(-0.01, 0.01)
        time_stamps.append(max(0.1, base_time + jitter))
    
    # Sort time stamps to ensure ordered submission
    time_stamps.sort()
    
    for request_time in time_stamps:
        # Get a random prompt from ShareGPT
        qa_pair = loader.get_random_qa()
        prompt = qa_pair.get('human', '')
        
        # Estimate input length from prompt (approximate: words * 1.3 for tokens)
        # Limit to reasonable range to avoid KV cache overflow
        estimated_tokens = int(len(prompt.split()) * 1.3)
        input_length = max(32, min(estimated_tokens, 256))  # Clamp between 32 and 256
        # Random output length between 50 and 256 tokens
        output_length = random.randint(50, 256)
        
        initial_requests.append((request_time, input_length, output_length))
    
    print(f"Generated {len(initial_requests)} burst requests for offline mode")
    # ------------------------------------------------------------------------------------------- #

    # ------------------------------------- Init System ------------------------------------ #
    host_ip: str = get_local_ip()
    assert host_ip.startswith("10"), "Local IP must be of form 10.xxx.xxx.xxx"
    llm_host.start_network_threads(CONFIG_BROADCAST_ADDR, host_ip, real_sys_config_file_name, scheduler_name)
    time.sleep(20)
    print("[Python] Cluster initialization finished!")
    # -------------------------------------------------------------------------------------- #
    ground_zero = time.time()
    next_query_id = 0
    flying_queries_dict = {}
    # -----  log items ----- #
    # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
    query_routes = []
    # time - query id - in/out - phase - context_len - this_iter_processed
    events = []
    # ---------------------- #
    last_log_time = 0
    while True:
        # get time
        now = time.time() - ground_zero
        if now > duration + 1:
            break

        if now - last_log_time > 1:
            print(f"[t={now}]")
            last_log_time = now

        # send out initial requests into the cluster
        while not len(initial_requests) == 0 and initial_requests[0][0] <= now:
            # the request has a time stamp smaller than now, should be sent
            expected_submit_time, input_length, output_length = initial_requests.pop(0)

            # get query id
            cur_query_id = next_query_id
            next_query_id += 1

            # send it into the cluster
            llm_host.launch_request(
                "prompt",  # request_type
                cur_query_id,  # request_id
                input_length,  # num_tokens
                input_length + output_length,  # max_num_tokens
                [i for i in range(input_length)],  # token_ids
                False,  # set_routing
                [],  # server_ids
                [],  # start_layer_ids
                [],  # end_layer_ids
            )

            # put into flying queries
            flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                            input_length=input_length,
                                                            output_length=output_length,
                                                            compute_node_uids=None,
                                                            start_layers=None,
                                                            end_layers=None,
                                                            pipeline=None)

            # time - query id - in/out - phase - context_len - this_iter_processed
            events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
            print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                  f"max_len = {input_length + output_length}")

        # get finished requests
        now = time.time() - ground_zero
        finished_query_ids, generated_token_ids, routes, num_layers = llm_host.gather_finished_requests()
        for query_uid, route_list, num_layer_list in zip(finished_query_ids, routes, num_layers):
            # first receive the message
            py_on_the_fly_query = flying_queries_dict[query_uid]
            if py_on_the_fly_query.processed_tokens == 0:
                # prompt phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "prompt", 0, py_on_the_fly_query.input_length + 1))
                py_on_the_fly_query.processed_tokens += py_on_the_fly_query.input_length + 1

                # now we can log the request with its route
                # query_id - input_len - output_len - compute_node_uids - start_layers - end_layers
                cur_log_start = 0
                start_layer_ids, end_layer_ids = [], []
                for num_layer in num_layer_list:
                    start_layer_ids.append(cur_log_start)
                    end_layer_ids.append(cur_log_start + num_layer)
                    cur_log_start += num_layer
                query_routes.append((query_uid, py_on_the_fly_query.input_length, py_on_the_fly_query.output_length,
                                     route_list, start_layer_ids, end_layer_ids))

            else:
                # decode phase
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "in", "decode", py_on_the_fly_query.processed_tokens, 1))
                py_on_the_fly_query.processed_tokens += 1

            # then we decide whether to send out new messages (decodes)
            max_size = py_on_the_fly_query.input_length + py_on_the_fly_query.output_length
            assert py_on_the_fly_query.processed_tokens <= max_size, "Found request that did not end!"
            if py_on_the_fly_query.processed_tokens == max_size:
                # not send: finished, remove from expectations
                del flying_queries_dict[query_uid]
                print(f"Query {query_uid}, finished (total_len={py_on_the_fly_query.processed_tokens})")

                # send a new query to replace the old one (using ShareGPT)
                qa_pair = loader.get_random_qa()
                prompt = qa_pair.get('human', '')
                estimated_tokens = int(len(prompt.split()) * 1.3)
                input_length = max(32, min(estimated_tokens, 256))  # Clamp between 32 and 256
                output_length = random.randint(50, 256)

                # get query id
                cur_query_id = next_query_id
                next_query_id += 1

                # send it into the cluster
                llm_host.launch_request(
                    "prompt",  # request_type
                    cur_query_id,  # request_id
                    input_length,  # num_tokens
                    input_length + output_length,  # max_num_tokens
                    [i for i in range(input_length)],  # token_ids
                    False,  # set_routing
                    [],  # server_ids
                    [],  # start_layer_ids
                    [],  # end_layer_ids
                )

                # put into flying queries
                flying_queries_dict[cur_query_id] = FlyingQuery(query_uid=cur_query_id,
                                                                input_length=input_length,
                                                                output_length=output_length,
                                                                compute_node_uids=None,
                                                                start_layers=None,
                                                                end_layers=None,
                                                                pipeline=None)

                # save log
                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, cur_query_id, "out", "prompt", 0, input_length + 1))
                print(f"Send out new query {cur_query_id}, input len = {input_length}, "
                      f"max_len = {input_length + output_length} (decode finish request replacement)")

            else:
                # then we send the query back into the cluster (next decode iter)
                llm_host.launch_request(
                    "decode",  # request_type
                    query_uid,  # request_id
                    py_on_the_fly_query.processed_tokens,  # num_tokens (context size)
                    max_size,  # max_num_tokens
                    [-1],  # token_ids
                    False,  # set_routing
                    [],  # server_ids
                    [],  # start_layer_ids
                    [],  # end_layer_ids
                )

                # time - query id - in/out - phase - context_len - this_iter_processed
                events.append((now, query_uid, "out", "decode", py_on_the_fly_query.processed_tokens, 1))

    # save logging files
    print(f"Queries still flying: {flying_queries_dict.keys()}.")
    query_routes_file_name = os.path.join(result_logging_dir, "query_route.txt")
    events_file_name = os.path.join(result_logging_dir, "events.txt")
    with open(query_routes_file_name, "w") as f:
        for item in query_routes:
            f.write(f"{item}\n")
    with open(events_file_name, "w") as f:
        for item in events:
            f.write(f"{item}\n")
