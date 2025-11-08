import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

# 设置专业风格的绘图参数
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'  # 使用无衬线字体
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.figsize'] = (16, 10)
mpl.rcParams['figure.dpi'] = 100

# 专业配色方案（来自Tableau）
BATCH_COLORS = {
    0: '#4E79A7',  # 深蓝
    1: '#F28E2B',  # 橙色
    2: '#E15759',  # 红色
    3: '#76B7B2',  # 青绿
    4: '#59A14F',  # 绿色
    5: '#EDC948',  # 黄色
    6: '#B07AA1',  # 紫色
    7: '#FF9DA7',  # 粉红
    8: '#9C755F',  # 棕色
    9: '#BAB0AC',  # 灰色,
    10: '#499894', # 青色
    11: '#A25050'  # 深红
}

# GPU专用颜色
GPU_COLORS = {
    0: '#1f77b4',  # GPU0 - 蓝色
    1: '#ff7f0e'   # GPU1 - 橙色
}

# 请求事件专用颜色
REQUEST_COLORS = {
    'added': '#2E8B57',  # 海绿色 - 请求到达
    'finished': '#DC143C'  # 深红色 - 请求完成
}

# 解析日志文件的函数 - 增强：支持GPU事件和汇总请求条目
def parse_log_file(filename, log_type):
    """
    返回:
      events: dict batch_id -> list of (event_type, timestamp, batch_size, stage)
      gpu_events: dict (gpu_id, server) -> list of (event_type, timestamp)
      request_events: dict timestamp -> list of tuples
    """
    events = defaultdict(list)
    gpu_events = defaultdict(list)  # 键为 (gpu_id, server)，值为事件列表
    request_events = defaultdict(list)
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # --- 匹配GPU事件: "GPU0 compute starts at 1761646689.3298755"
            gpu_start_match = re.match(r'GPU(\d+)\s+compute starts at\s+([\d.]+)$', line)
            gpu_end_match = re.match(r'GPU(\d+)\s+compute ends at\s+([\d.]+)$', line)
            
            if gpu_start_match:
                gpu_id = int(gpu_start_match.group(1))
                timestamp = float(gpu_start_match.group(2))
                gpu_events[(gpu_id, log_type)].append(('gpu_start', timestamp))
                continue
                
            if gpu_end_match:
                gpu_id = int(gpu_end_match.group(1))
                timestamp = float(gpu_end_match.group(2))
                gpu_events[(gpu_id, log_type)].append(('gpu_end', timestamp))
                continue

            # --- 匹配汇总请求条目
            m_req_count = re.match(r'(\d+)\s+requests\s+(finished|added)\s+at\s+([\d.]+)$', line)
            if m_req_count:
                count = int(m_req_count.group(1))
                action = m_req_count.group(2)
                ts = float(m_req_count.group(3))
                key = f'{action}_count'
                request_events[ts].append((key, count))
                continue

            # --- 兼容原先的逐条 request 日志
            request_added_match = re.match(r'request\s+([a-f0-9]+)\s+is added at\s+([\d.]+)$', line)
            request_finished_match = re.match(r'request\s+([a-f0-9]+)\s+finished at\s+([\d.]+)$', line)
            if request_added_match:
                request_id = request_added_match.group(1)
                timestamp = float(request_added_match.group(2))
                request_events[timestamp].append(('added', request_id))
                continue
            if request_finished_match:
                request_id = request_finished_match.group(1)
                timestamp = float(request_finished_match.group(2))
                request_events[timestamp].append(('finished', request_id))
                continue

            # --- 匹配常规事件行
            match = re.match(r'(\d+)(?:\s+(\d+))?\s+(.+?)\s+at\s+([\d.]+)$', line)
            if not match:
                continue
                
            batch_id = int(match.group(1))
            batch_size = match.group(2)
            event_desc = match.group(3).strip()
            timestamp = float(match.group(4))

            stage_match = re.search(r'\(([^)]+)\)', event_desc)
            stage = stage_match.group(1) if stage_match else None

            if batch_size:
                batch_size = int(batch_size)
            else:
                size_match = re.search(r'(\d+)\s+compute starts', event_desc)
                if size_match:
                    batch_size = int(size_match.group(1))

            event_type = None
            if "compute starts" in event_desc:
                event_type = 'compute_start'
            elif "compute ends" in event_desc:
                event_type = 'compute_end'
            elif "trans starts" in event_desc:
                event_type = 'trans_start'
            elif "trans ends" in event_desc:
                event_type = 'trans_end'
            elif "serialization starts" in event_desc:
                event_type = 'serial_start'
            elif "serialization ends" in event_desc:
                event_type = 'serial_end'
            elif "back to head" in event_desc:
                event_type = 'back_to_head'
            elif "recv" in event_desc:
                event_type = 'recv'
            else:
                continue

            events[batch_id].append((event_type, timestamp, batch_size or 0, stage))
            
    return events, gpu_events, request_events

# 解析日志文件
print("解析server1日志...")
server1_events, server1_gpu_events, server1_requests = parse_log_file('./heter/qwen3-32b-awq/server0.log', 'server1')
print("解析server2日志...")
server2_events, server2_gpu_events, server2_requests = parse_log_file('./heter/qwen3-32b-awq/server1.log', 'server2')

# 提取所有时间戳用于归一化
all_timestamps = []
for batch_id, events in server1_events.items():
    print(f"Server1 Batch {batch_id} 事件:")
    for event_type, ts, size, stage in events:
        print(f"  {event_type} ({stage}) at {ts}")
        all_timestamps.append(ts)

for batch_id, events in server2_events.items():
    print(f"Server2 Batch {batch_id} 事件:")
    for event_type, ts, size, stage in events:
        print(f"  {event_type} ({stage}) at {ts}")
        all_timestamps.append(ts)

# 添加GPU事件的时间戳
for (gpu_id, server), events in server1_gpu_events.items():
    for event_type, ts in events:
        print(f"Server1 GPU{gpu_id} {event_type} at {ts}")
        all_timestamps.append(ts)

for (gpu_id, server), events in server2_gpu_events.items():
    for event_type, ts in events:
        print(f"Server2 GPU{gpu_id} {event_type} at {ts}")
        all_timestamps.append(ts)

# 添加请求事件的时间戳
for ts, events in server1_requests.items():
    for tup in events:
        print(f"Server1 Request event {tup} at {ts}")
        all_timestamps.append(ts)

for ts, events in server2_requests.items():
    for tup in events:
        print(f"Server2 Request event {tup} at {ts}")
        all_timestamps.append(ts)

if not all_timestamps:
    print("错误: 没有解析到任何事件!")
    exit(1)

base_time = min(all_timestamps)
print(f"基准时间: {base_time}")

# 归一化时间戳函数
def normalize(ts):
    return ts - base_time

# 创建绘图
fig, ax = plt.subplots(figsize=(16, 10))
fig.set_facecolor('#F8F9FA')
ax.set_facecolor('#FFFFFF')

# 资源层级定义 - 只保留GPU部分
RESOURCE_LEVELS = {
    'Server1 GPU0': 6.0,
    'Server1 GPU1': 5.0,
    'Network Transfer': 3.5,
    'Server2 GPU0': 2.5,
    'Server2 GPU1': 1.5
}

# Batch标注位置 - 在Server1 GPU1下方
BATCH_ANNOTATION_LEVEL = 4.0

# 请求事件位置
REQUEST_LEVEL = 7.0

# 线宽和透明度设置
GPU_BAR_HEIGHT = 0.3
TRANSFER_BAR_HEIGHT = 0.2
ALPHA = 0.9

# 设置要显示的时间范围
start_time = 1762164832.4847505
end_time = 1762164839.4847505
# 归一化时间范围
norm_start = normalize(start_time)
norm_end = normalize(end_time)
print(f"显示时间范围: {norm_start:.6f} 到 {norm_end:.6f} (相对基准时间)")

# 存储批注对象用于调整位置
annotations = []
legend_handles = []
found_transfers = []

# 绘制GPU计算段函数
def draw_gpu_segments(gpu_events, server_name):
    for (gpu_id, server), events in gpu_events.items():
        if server != server_name:
            continue
            
        # 分离开始和结束事件
        starts = [ts for etype, ts in events if etype == 'gpu_start']
        ends = [ts for etype, ts in events if etype == 'gpu_end']
        
        starts.sort()
        ends.sort()
        
        # 配对开始和结束事件
        min_count = min(len(starts), len(ends))
        if len(starts) != len(ends):
            print(f"警告: {server_name} GPU{gpu_id} 的开始({len(starts)})和结束({len(ends)})事件数量不匹配")
            starts = starts[:min_count]
            ends = ends[:min_count]
        
        for start, end in zip(starts, ends):
            norm_start_ts = normalize(start)
            norm_end_ts = normalize(end)
            
            # 检查是否在时间范围内
            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                
                # 计算持续时间
                duration_ms = (end - start) * 1000
                
                # 选择颜色
                color = GPU_COLORS.get(gpu_id, '#1f77b4')
                
                # 使用正确的键名格式
                resource_key = f'Server{server_name[-1]} GPU{gpu_id}'
                if resource_key not in RESOURCE_LEVELS:
                    print(f"警告: 资源键 {resource_key} 不在 RESOURCE_LEVELS 中")
                    continue
                    
                resource_level = RESOURCE_LEVELS[resource_key]
                
                # 绘制GPU计算段
                bar_height = GPU_BAR_HEIGHT
                y_bottom = resource_level - bar_height / 2.0
                
                rect = Rectangle(
                    (display_start, y_bottom),
                    display_end - display_start,
                    bar_height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=ALPHA,
                    linewidth=0.5,
                    zorder=2
                )
                ax.add_patch(rect)
                
                # 添加图例句柄（只添加一次）
                gpu_key = f'gpu{gpu_id}_{server_name}'
                if gpu_key not in [h[0] for h in legend_handles if isinstance(h[0], str)]:
                    legend_handles.append((gpu_key, plt.Line2D([0], [0], color=color, lw=4, alpha=ALPHA)))
                
                # 添加文本标注
                mid_x = (display_start + display_end) / 2
                text = f"GPU{gpu_id}\n{duration_ms:.1f}ms"
                
                # 根据服务器决定标注位置
                if server_name == 'server1':
                    ann_y = resource_level - bar_height/2 - 0.15
                else:
                    ann_y = resource_level + bar_height/2 + 0.15
                
                ann = ax.text(
                    mid_x, ann_y, text,
                    ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2')
                )
                annotations.append(ann)

# 绘制请求事件函数
def draw_request_events(request_events):
    request_counts = defaultdict(lambda: defaultdict(int))
    
    for ts, events in request_events.items():
        for ev in events:
            etype = ev[0]
            if etype.endswith('_count'):
                action = etype.replace('_count', '')
                count = ev[1]
                request_counts[ts][action] += int(count)
            else:
                action = etype
                request_counts[ts][action] += 1
    
    for ts, event_counts in request_counts.items():
        norm_ts = normalize(ts)
        
        if norm_start <= norm_ts <= norm_end:
            for event_type, count in event_counts.items():
                color = REQUEST_COLORS.get(event_type, '#000000')
                event_name = "arrives" if event_type == "added" else "finishes"
                
                # 绘制箭头指向Server1 GPU0
                arrow = FancyArrowPatch(
                    (norm_ts, REQUEST_LEVEL), (norm_ts, RESOURCE_LEVELS['Server1 GPU0'] + 0.3),
                    arrowstyle='->', mutation_scale=15, color=color, linewidth=2, alpha=0.8
                )
                ax.add_patch(arrow)
                
                # 添加文本标注
                text = f"{count} req {event_name}"
                ann = ax.text(
                    norm_ts, REQUEST_LEVEL + 0.15, text,
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, boxstyle='round,pad=0.3')
                )
                annotations.append(ann)
                
                if event_type not in [h[0] for h in legend_handles if isinstance(h[0], str)]:
                    legend_handles.append((event_type, plt.Line2D([0], [0], color=color, lw=3, marker='o', markersize=8)))

# 绘制batch标注函数 - 沿用原来server compute部分的标注逻辑
def draw_batch_annotations():
    # 收集所有batch的计算段信息
    for batch_id, batch_events in server1_events.items():
        compute_starts = [(ts, size, stage) for etype, ts, size, stage in batch_events if etype == 'compute_start']
        compute_ends = [(ts, stage) for etype, ts, _, stage in batch_events if etype == 'compute_end']
        
        compute_starts.sort(key=lambda x: x[0])
        compute_ends.sort(key=lambda x: x[0])
        
        min_count = min(len(compute_starts), len(compute_ends))
        if len(compute_starts) != len(compute_ends):
            print(f"警告: Batch {batch_id} 的计算开始({len(compute_starts)})和结束({len(compute_ends)})事件数量不匹配，取最小值 {min_count} 进行匹配")
            compute_starts = compute_starts[:min_count]
            compute_ends = compute_ends[:min_count]
        
        for (start, batch_size, start_stage), (end, end_stage) in zip(compute_starts, compute_ends):
            stage = start_stage or end_stage
            norm_start_ts = normalize(start)
            norm_end_ts = normalize(end)
            
            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                
                duration_ms = (end - start) * 1000
                
                color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                
                # 在batch标注层级添加标注
                mid_x = (display_start + display_end) / 2
                text = f"Batch {batch_id}\nbs={batch_size}\n\n{stage or ''}"
                
                ann = ax.text(
                    mid_x, BATCH_ANNOTATION_LEVEL, text,
                    ha='center', va='center', fontsize=9,
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.3')
                )
                annotations.append(ann)
                
                if batch_id not in [h[0] for h in legend_handles if isinstance(h[0], int)]:
                    legend_handles.append((batch_id, plt.Line2D([0], [0], color=color, lw=4, alpha=ALPHA)))
    
    # 也处理server2中的batch，避免遗漏
    for batch_id, batch_events in server2_events.items():
        if batch_id in server1_events:  # 如果已经在server1中处理过，跳过
            continue
            
        compute_starts = [(ts, size, stage) for etype, ts, size, stage in batch_events if etype == 'compute_start']
        compute_ends = [(ts, stage) for etype, ts, _, stage in batch_events if etype == 'compute_end']
        
        compute_starts.sort(key=lambda x: x[0])
        compute_ends.sort(key=lambda x: x[0])
        
        min_count = min(len(compute_starts), len(compute_ends))
        if len(compute_starts) != len(compute_ends):
            compute_starts = compute_starts[:min_count]
            compute_ends = compute_ends[:min_count]
        
        for (start, batch_size, start_stage), (end, end_stage) in zip(compute_starts, compute_ends):
            stage = start_stage or end_stage
            norm_start_ts = normalize(start)
            norm_end_ts = normalize(end)
            
            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                
                duration_ms = (end - start) * 1000
                
                color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                
                # 在batch标注层级添加标注
                mid_x = (display_start + display_end) / 2
                text = f"Batch {batch_id}\nbs={batch_size}\n{duration_ms:.1f}ms\n{stage or ''}"
                
                ann = ax.text(
                    mid_x, BATCH_ANNOTATION_LEVEL, text,
                    ha='center', va='center', fontsize=9,
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.3')
                )
                annotations.append(ann)
                
                if batch_id not in [h[0] for h in legend_handles if isinstance(h[0], int)]:
                    legend_handles.append((batch_id, plt.Line2D([0], [0], color=color, lw=4, alpha=ALPHA)))

# 绘制传输段函数
def draw_transfer_segments():
    global found_transfers
    
    print("\n绘制中间结果传输 (server1 -> server2): trans_start to recv")
    for batch_id in set(list(server1_events.keys()) + list(server2_events.keys())):
        s1_trans_starts = [ts for etype, ts, _, _ in server1_events.get(batch_id, []) 
                          if etype == 'trans_start']
        s1_trans_starts.sort()
        
        s2_recvs = [ts for etype, ts, _, _ in server2_events.get(batch_id, []) 
                   if etype == 'recv']
        s2_recvs.sort()
        
        print(f"Batch {batch_id}: trans_starts={len(s1_trans_starts)}, recvs={len(s2_recvs)}")
        
        if s1_trans_starts and s2_recvs:
            for i in range(min(len(s1_trans_starts), len(s2_recvs))):
                start = s1_trans_starts[i]
                end = s2_recvs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                print(f"  中间传输 {i+1}: {start} -> {end} (原始时间)")
                print(f"  归一化后: {norm_start_ts:.6f} -> {norm_end_ts:.6f}")
                
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    duration_ms = (end - start) * 1000
                    
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    print(f"  在时间范围内! 绘制线段: {display_start:.6f} -> {display_end:.6f}")
                    
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network Transfer'] - bar_height / 2.0
                    rect = Rectangle(
                        (display_start, y_bottom),
                        display_end - display_start,
                        bar_height,
                        facecolor=color,
                        edgecolor=color,
                        alpha=ALPHA,
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(rect)
                    
                    mid_x = (display_start + display_end) / 2
                    text = f"To Server2\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Transfer'] - bar_height/2 - 0.12
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("intermediate", batch_id, start, end))
    
    print("\n绘制结果返回传输 (server2 -> server1): trans_start to back_to_head")
    for batch_id in set(list(server1_events.keys()) + list(server2_events.keys())):
        s2_trans_starts = [ts for etype, ts, _, _ in server2_events.get(batch_id, []) 
                          if etype == 'trans_start']
        s2_trans_starts.sort()
        
        s1_backs = [ts for etype, ts, _, _ in server1_events.get(batch_id, []) 
                   if etype == 'back_to_head']
        s1_backs.sort()
        
        print(f"Batch {batch_id}: trans_starts={len(s2_trans_starts)}, back_to_heads={len(s1_backs)}")
        
        if s2_trans_starts and s1_backs:
            for i in range(min(len(s2_trans_starts), len(s1_backs))):
                start = s2_trans_starts[i]
                end = s1_backs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                print(f"  结果传输 {i+1}: {start} -> {end} (原始时间)")
                print(f"  归一化后: {norm_start_ts:.6f} -> {norm_end_ts:.6f}")
                
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    duration_ms = (end - start) * 1000
                    
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    print(f"  在时间范围内! 绘制线段: {display_start:.6f} -> {display_end:.6f}")
                    
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network Transfer'] - bar_height / 2.0
                    rect = Rectangle(
                        (display_start, y_bottom),
                        display_end - display_start,
                        bar_height,
                        facecolor=color,
                        edgecolor=color,
                        alpha=ALPHA * 0.85,
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(rect)
                    
                    mid_x = (display_start + display_end) / 2
                    text = f"To Server1\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Transfer'] - bar_height/2 - 0.12
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("result", batch_id, start, end))

# 绘制所有部分
print("\n绘制Server1 GPU段...")
draw_gpu_segments(server1_gpu_events, 'server1')
print("绘制Server2 GPU段...")
draw_gpu_segments(server2_gpu_events, 'server2')
print("绘制网络传输段...")
draw_transfer_segments()
print("绘制请求事件...")
draw_request_events(server1_requests)
print("绘制Batch标注...")
draw_batch_annotations()

if not found_transfers:
    print("警告: 没有找到任何网络传输事件!")
else:
    print(f"成功绘制 {len(found_transfers)} 个传输事件")

# 设置图表属性
ax.set_yticks(list(RESOURCE_LEVELS.values()) + [REQUEST_LEVEL, BATCH_ANNOTATION_LEVEL])
ax.set_yticklabels(list(RESOURCE_LEVELS.keys()) + ['Request Events', 'Batch Annotations'], fontsize=11, fontweight='bold')
ax.set_ylabel('Resources', fontsize=12, fontweight='bold')
ax.set_xlabel(f'Time (seconds from base time {base_time:.6f})', fontsize=11)
ax.set_title(f'Distributed Computing Timeline with GPU Activities', fontsize=14, fontweight='bold', pad=15)

# 设置x轴范围
ax.set_xlim(norm_start, norm_end)

# 设置y轴范围以包含所有层级
ax.set_ylim(1.0, 7.5)

# 添加时间刻度线
ax.grid(True, axis='x', linestyle='--', alpha=0.6)

# 添加资源分隔线
for y in RESOURCE_LEVELS.values():
    ax.axhline(y=y, color='gray', alpha=0.3, linewidth=0.5)

# 添加请求事件水平线
ax.axhline(y=REQUEST_LEVEL, color='gray', alpha=0.3, linewidth=0.5, linestyle=':')

# 添加Batch标注水平线
ax.axhline(y=BATCH_ANNOTATION_LEVEL, color='gray', alpha=0.3, linewidth=0.5, linestyle=':')

# 创建专业图例
if legend_handles:
    # 分离不同类型的图例
    batch_handles = [h[1] for h in legend_handles if isinstance(h[0], int)]
    batch_labels = [f'Batch {h[0]}' for h in legend_handles if isinstance(h[0], int)]
    
    gpu_handles = [h[1] for h in legend_handles if isinstance(h[0], str) and h[0].startswith('gpu')]
    gpu_labels = []
    for h in legend_handles:
        if isinstance(h[0], str) and h[0].startswith('gpu'):
            parts = h[0].split('_')
            gpu_id = parts[0][3:]  # 提取GPU编号
            server = parts[1].title()  # 提取服务器名并首字母大写
            gpu_labels.append(f'{server} GPU{gpu_id}')
    
    request_handles = [h[1] for h in legend_handles if isinstance(h[0], str) and not h[0].startswith('gpu')]
    request_labels = [f'Request {h[0].title()}' for h in legend_handles if isinstance(h[0], str) and not h[0].startswith('gpu')]
    
    # Batch颜色图例
    if batch_handles:
        batch_legend = plt.legend(
            batch_handles,
            batch_labels,
            title='Batch ID',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(batch_legend)
    
    # GPU图例
    if gpu_handles:
        gpu_legend = plt.legend(
            gpu_handles,
            gpu_labels,
            title='GPU Activities',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.85),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(gpu_legend)
    
    # 请求事件图例
    if request_handles:
        request_legend = plt.legend(
            request_handles,
            request_labels,
            title='Request Events',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.70),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(request_legend)

# 添加时间范围标记
ax.text(
    0.5, -0.10, 
    f'Time Range: {start_time:.6f} - {end_time:.6f} | Base Time: {base_time:.6f}',
    transform=ax.transAxes,
    ha='center',
    fontsize=9,
    color='#555555'
)

# 添加边框
for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')
    spine.set_linewidth=0.8

# 调整标注位置避免重叠
def adjust_annotations(annotations, min_distance=0.02):
    annotations.sort(key=lambda ann: ann.get_position()[0])
    
    for i in range(1, len(annotations)):
        prev = annotations[i-1]
        curr = annotations[i]
        prev_pos = prev.get_position()
        curr_pos = curr.get_position()
        
        if abs(curr_pos[0] - prev_pos[0]) < min_distance:
            offset = min_distance - abs(curr_pos[0] - prev_pos[0])
            new_y = curr_pos[1] - offset * 0.5
            curr.set_position((curr_pos[0], new_y))

# 应用标注调整
if annotations:
    adjust_annotations(annotations)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.92, left=0.08, right=0.95)

# 保存图像（可选）
# plt.savefig('distributed_timeline_with_gpu.pdf', bbox_inches='tight', dpi=300)
plt.show()