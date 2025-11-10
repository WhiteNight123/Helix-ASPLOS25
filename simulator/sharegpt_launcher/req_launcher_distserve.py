#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import json
import time
import random
import sys
from typing import Optional, List
from sharegpt_loader_distserve import ShareGPTLoader
from dataset_sequences import sequence_shuffled
from token_counts import input_tokens_list, output_tokens_list

def start_poisson_for_duration(
    url: str,
    loader: ShareGPTLoader,
    avg_rps: float,
    duration_seconds: float,
    sequence_shuffled: List[int],
    input_tokens_list: List[int],
    output_tokens_list: List[int],
    max_concurrency: Optional[int] = None,
    verbose: bool = True,
):
    """
    按泊松到达（指数分布间隔）在指定持续时间内提交请求。
    - url: curl 请求 URL（字符串）
    - loader: ShareGPTLoader实例
    - avg_rps: 每秒平均请求数 λ（requests/second），必须 > 0
    - duration_seconds: 持续时间（秒），脚本在发送到达持续时间到时立即返回（不等待子进程）
    - sequence_shuffled: 顺序抽取的索引列表
    - input_tokens_list: 输入token数量列表
    - output_tokens_list: 输出token数量列表
    - max_concurrency: 可选并发上限（同时存在的子进程数），None 表示不限制
    - verbose: 是否打印简短提交进度
    """
    if avg_rps <= 0:
        raise ValueError("avg_rps 必须 > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds 必须 > 0")
    if max_concurrency is not None and max_concurrency <= 0:
        raise ValueError("max_concurrency 必须为正整数或 None")
    
    if len(sequence_shuffled) != len(input_tokens_list) or len(sequence_shuffled) != len(output_tokens_list):
        raise ValueError("sequence_shuffled, input_tokens_list, output_tokens_list 长度必须相同")
    
    dn = subprocess.DEVNULL  # 跨平台的空输出

    start_time = time.time()
    end_time = start_time + duration_seconds
    submits = 0
    processes = []  # 存放当前仍未被清理的 Popen 对象
    sequence_index = 0  # 当前在sequence_shuffled中的位置

    try:
        while True:
            # 按照sequence_shuffled顺序获取QA对
            if sequence_index >= len(sequence_shuffled):
                print("警告: sequence_shuffled已用完，从头开始循环")
                sequence_index = 0
                
            qa_index = sequence_shuffled[sequence_index]
            qa_pair = loader.get_qa_by_index(qa_index)
            
            if not qa_pair:
                print(f"错误: 无法获取索引 {qa_index} 的QA对")
                sequence_index += 1
                continue
            
            prompt = qa_pair.get('human')
            input_tokens = input_tokens_list[sequence_index]
            output_tokens = output_tokens_list[sequence_index]
            
            data = {
                "prompt": prompt,
                "min_tokens": output_tokens,
                "max_tokens": output_tokens,
                "temperature": 0
            }
            data_str = json.dumps(data)
            now = time.time()
            if now >= end_time:
                break

            # 清理已经结束的子进程（非阻塞）
            if processes:
                alive = []
                for p in processes:
                    if p.poll() is None:
                        alive.append(p)
                processes = alive

            # 如果设置了 max_concurrency，且达到上限，则短暂等待并重试（不阻塞任何单个进程）
            if max_concurrency is not None and len(processes) >= max_concurrency:
                # 为避免忙等，睡一小段时间后继续检查
                time.sleep(0.001)
                continue

            # 构造 curl 命令
            command = [
                "curl", "-sS", url,
                "-H", "Content-Type: application/json",
                "-d", data_str
            ]

            # 启动子进程，不等待完成，输出重定向到 DEVNULL
            try:
                p = subprocess.Popen(command, stdout=dn, stderr=dn, close_fds=True)
                processes.append(p)
                submits += 1
                if verbose:
                    elapsed = now - start_time
                    print(f"[提交] #{submits} (t={elapsed:.3f}s) 索引={qa_index} 输入长度={input_tokens} 输出长度={output_tokens} 当前并发={len(processes)}")
            except Exception as e:
                # 启动失败则记录并继续
                if verbose:
                    print(f"[ERROR] 启动请求失败: {e}", file=sys.stderr)

            sequence_index += 1  # 移动到下一个索引

            # 计算下一到达间隔（指数分布），参数 λ = avg_rps
            wait_seconds = random.expovariate(avg_rps)

            # 若下一个到达时间会超出结束时间，则不再等待并退出发送循环
            if time.time() + wait_seconds >= end_time:
                break

            # 为避免极短的 sleep 导致 busy-loop，我们对最小睡眠进行微小下限
            if wait_seconds < 0.0005:
                time.sleep(0.0005)
            else:
                time.sleep(wait_seconds)

    except KeyboardInterrupt:
        print("\n用户中断，停止提交。", file=sys.stderr)

    total_elapsed = time.time() - start_time
    print(f"提交窗口已结束（实际提交耗时 {total_elapsed:.3f} 秒），共提交请求数: {submits}")
    print("脚本退出：不等待子进程完成。")

if __name__ == "__main__":
    default_url = "http://172.17.0.8:20000/generate"

    try:
        print("按泊松到达（以 avg_rps 控制平均每秒请求数）持续提交请求脚本")

        url = default_url

        while True:
            s = input("每秒平均请求数 avg_rps（requests/second，>0，例如 5 或 0.5）: ").strip()
            try:
                avg_rps = float(s)
                if avg_rps <= 0:
                    raise ValueError
                break
            except Exception:
                print("请输入一个大于 0 的数值。")

        while True:
            s = input("持续时间（秒，例如 30）: ").strip()
            try:
                duration = float(s)
                if duration <= 0:
                    raise ValueError
                break
            except Exception:
                print("请输入一个大于 0 的数值（秒）。")


        print(f"使用序列长度: {len(sequence_shuffled)}")
        print(f"输入token列表长度: {len(input_tokens_list)}")
        print(f"输出token列表长度: {len(output_tokens_list)}")

        s = False
        max_concurrency = None
        if s:
            try:
                m = int(s)
                if m <= 0:
                    raise ValueError
                max_concurrency = m
            except Exception:
                print("max_concurrency 非法，使用不限制。")
                max_concurrency = None

        s = 'y'
        verbose = s.startswith("y")

        loader = ShareGPTLoader("/mnt/lvm-data/home/dataset/sharegpt/common_en_70k.jsonl")
        print('正在加载shareGPT数据集到内存...')
        loader.load_data()
        print('加载完毕，即将开始测试')

        print("准备开始提交（按 Ctrl+C 可提前结束）...")
        start_poisson_for_duration(
            url=url,
            loader=loader,
            avg_rps=avg_rps,
            duration_seconds=duration,
            sequence_shuffled=sequence_shuffled,
            input_tokens_list=input_tokens_list,
            output_tokens_list=output_tokens_list,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )

    except KeyboardInterrupt:
        print("\n用户中断，退出。")
        sys.exit(0)