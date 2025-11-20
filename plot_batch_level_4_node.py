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
    9: '#BAB0AC',  # 灰色
    10: '#499894', # 青色
    11: '#A25050', # 深红
    12: '#7B4F9D', # 紫色
    13: '#C44E52', # 红色
    14: '#55A868', # 绿色
    15: '#8172B2'  # 蓝紫色
}

# 请求事件专用颜色
REQUEST_COLORS = {
    'added': '#2E8B57',  # 海绿色 - 请求到达
    'finished': '#DC143C'  # 深红色 - 请求完成
}

# 服务器颜色映射
SERVER_COLORS = {
    'server1': '#4E79A7',
    'server2': '#F28E2B', 
    'server3': '#E15759',
    'server4': '#76B7B2'
}

# 解析日志文件的函数 - 增强：支持完整的批次列表匹配
def parse_log_file(filename, log_type):
    """
    返回:
      events: dict batch_key -> list of (event_type, timestamp, batch_size, stage)
              batch_key: 完整的批次标识，如 "0 [16, 17, 2, 3]"
              event_type: 'compute_start','compute_end','trans_start','trans_end','serial_start','serial_end','back_to_head','recv'
              stage: None or string like 'prefill'/'decode' 等
      request_events: dict timestamp -> list of tuples:
              - ('added', req_id) or ('finished', req_id)  (原先逐条)
              - ('added_count', count) or ('finished_count', count) (汇总条目)
    """
    events = defaultdict(list)
    request_events = defaultdict(list)  # 专门存储请求事件
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # --- 新增：匹配汇总请求条目，例如 "1 requests finished at 1761024545.164717"
            m_req_count = re.match(r'(\d+)\s+requests\s+(finished|added)\s+at\s+([\d.]+)$', line)
            if m_req_count:
                count = int(m_req_count.group(1))
                action = m_req_count.group(2)  # 'finished' 或 'added'
                ts = float(m_req_count.group(3))
                key = f'{action}_count'  # 'finished_count' 或 'added_count'
                request_events[ts].append((key, count))
                continue

            # --- 兼容原先的逐条 request 日志: "request <id> is added at <ts>" 或 "request <id> finished at <ts>"
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

            # --- 匹配常规事件行: 支持完整的批次列表匹配
            # 例如: "0 [16, 17, 2, 3] compute starts (prefill) at 1761024538.0907214"
            # 或者: "0 compute starts (prefill) at 1761024538.0907214" (无列表的旧格式)
            match = re.match(r'(\d+)(?:\s+(\[[\d\s,]*\]))?(?:\s+(\d+))?\s+(.+?)\s+at\s+([\d.]+)$', line)
            if not match:
                continue
                
            batch_id = int(match.group(1))
            batch_list = match.group(2)  # 完整的列表，如 "[16, 17, 2, 3]"
            batch_size_alt = match.group(3)  # 备用的batch_size（旧格式）
            event_desc = match.group(4).strip()
            timestamp = float(match.group(5))

            # 创建批次键：如果存在完整列表，使用"id [list]"格式，否则只用id
            if batch_list:
                batch_key = f"{batch_id} {batch_list}"
            else:
                batch_key = str(batch_id)

            # 尝试提取阶段注释，例如 "compute starts (prefill)" -> stage='prefill'
            stage_match = re.search(r'\(([^)]+)\)', event_desc)
            stage = stage_match.group(1) if stage_match else None

            # 规范化 batch_size
            batch_size = 0
            if batch_list:
                # 从列表中提取batch_size（列表长度）
                try:
                    # 解析列表字符串为Python列表
                    batch_list_parsed = eval(batch_list)
                    batch_size = len(batch_list_parsed)
                except:
                    batch_size = 0
            elif batch_size_alt:
                batch_size = int(batch_size_alt)
            else:
                # 尝试从事件描述中提取 batch size（若像 "8 compute starts" 的情况）
                size_match = re.search(r'(\d+)\s+compute starts', event_desc)
                if size_match:
                    batch_size = int(size_match.group(1))

            # 提取事件类型
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
                continue  # 忽略未知事件类型

            # 存储为四元组，stage 可能为 None
            events[batch_key].append((event_type, timestamp, batch_size, stage))
            
    return events, request_events

# 解析4个服务器的日志文件
print("解析server1日志...")
server1_events, server1_requests = parse_log_file('/root/Helix-ASPLOS25/examples/real_sys/log/measurement_11.log', 'server1')
print("解析server2日志...")
server2_events, server2_requests = parse_log_file('/root/Helix-ASPLOS25/examples/real_sys/log/measurement_12.log', 'server2')
print("解析server3日志...")
server3_events, server3_requests = parse_log_file('/root/Helix-ASPLOS25/examples/real_sys/log/measurement_13.log', 'server3')
print("解析server4日志...")
server4_events, server4_requests = parse_log_file('/root/Helix-ASPLOS25/examples/real_sys/log/measurement_14.log', 'server4')

# 提取所有时间戳用于归一化
all_timestamps = []
servers_events = [server1_events, server2_events, server3_events, server4_events]
servers_requests = [server1_requests, server2_requests, server3_requests, server4_requests]
server_names = ['server1', 'server2', 'server3', 'server4']

for i, (events, name) in enumerate(zip(servers_events, server_names)):
    print(f"\n{name.upper()} 事件:")
    for batch_key, batch_events in events.items():
        print(f"  Batch {batch_key}: {len(batch_events)} 个事件")
        for event_type, ts, size, stage in batch_events:
            all_timestamps.append(ts)

# 添加请求事件的时间戳
for i, (requests, name) in enumerate(zip(servers_requests, server_names)):
    for ts, events in requests.items():
        for tup in events:
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
fig, ax = plt.subplots(figsize=(18, 12))
fig.set_facecolor('#F8F9FA')  # 浅灰色背景
ax.set_facecolor('#FFFFFF')   # 白色轴背景

# 资源层级定义 - 为4个服务器调整间距
RESOURCE_LEVELS = {
    'Request Events': 8.0,
    'Server1 Compute': 6.0,
    'Network S1→S2': 4.5,
    'Server2 Compute': 3.0,
    'Network S2→S3': 1.5,
    'Server3 Compute': 0.0,
    'Network S3→S4': -1.5,
    'Server4 Compute': -3.0,
    'Network Returns': -4.5  # 返回传输统一层
}

# 线宽和透明度设置
COMPUTE_BAR_HEIGHT = 0.3  # 计算条高度（y轴单位）
TRANSFER_BAR_HEIGHT = 0.2  # 传输条高度（y轴单位）
ALPHA = 0.9

# 设置要显示的时间范围（根据实际数据调整）
start_time = 1763548560.238678
end_time = 1763548560.938678

# 归一化时间范围
norm_start = normalize(start_time)
norm_end = normalize(end_time)
print(f"显示时间范围: {norm_start:.6f} 到 {norm_end:.6f} (相对基准时间)")

# 存储批注对象用于调整位置
annotations = []
legend_handles = []  # 用于自定义图例
found_transfers = []  # 存储找到的传输事件

# 绘制请求事件函数 - 支持汇总数量条目
def draw_request_events(request_events):
    request_counts = defaultdict(lambda: defaultdict(int))  # 按时间戳和事件类型计数
    
    # 统计每个时间戳的事件数量
    for ts, events in request_events.items():
        for ev in events:
            # ev 可能是 ('added', req_id) 或 ('finished', req_id) 或 ('added_count', count) ...
            etype = ev[0]
            if etype.endswith('_count'):
                # ('finished_count', count)
                action = etype.replace('_count', '')
                count = ev[1]
                request_counts[ts][action] += int(count)
            else:
                # ('added', req_id) 或 ('finished', req_id)
                action = etype
                request_counts[ts][action] += 1
    
    # 绘制事件
    for ts, event_counts in request_events.items():
        norm_ts = normalize(ts)
        
        # 检查是否在时间范围内
        if norm_start <= norm_ts <= norm_end:
            for event_type, count in event_counts.items():
                color = REQUEST_COLORS.get(event_type, '#000000')
                event_name = "arrives" if event_type == "added" else "finishes"
                
                # 绘制箭头指向Server1 Compute
                arrow = FancyArrowPatch(
                    (norm_ts, RESOURCE_LEVELS['Request Events']), 
                    (norm_ts, RESOURCE_LEVELS['Server1 Compute'] + 0.2),
                    arrowstyle='->', mutation_scale=15, color=color, linewidth=2, alpha=0.8
                )
                ax.add_patch(arrow)
                
                # 添加文本标注
                text = f"{count} req {event_name}"
                ann = ax.text(
                    norm_ts, RESOURCE_LEVELS['Request Events'] + 0.15, text,
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, boxstyle='round,pad=0.3')
                )
                annotations.append(ann)
                
                # 添加图例句柄（只添加一次）
                if event_type not in [h[0] for h in legend_handles if isinstance(h[0], str)]:
                    legend_handles.append((event_type, plt.Line2D([0], [0], color=color, lw=3, marker='o', markersize=8)))

# 绘制计算段函数 - 支持阶段并用 hatch 区分
def draw_compute_segments(events, resource_level, server_name):
    for batch_key, batch_events in events.items():
        # 从batch_key中提取batch_id用于颜色映射
        try:
            batch_id = int(batch_key.split()[0])  # 提取开头的数字作为batch_id
        except:
            batch_id = hash(batch_key) % len(BATCH_COLORS)  # 如果解析失败，使用哈希
        
        # 收集所有计算开始和结束事件，按阶段配对
        compute_starts = [(ts, size, stage) for etype, ts, size, stage in batch_events if etype == 'compute_start']
        compute_ends = [(ts, stage) for etype, ts, _, stage in batch_events if etype == 'compute_end']
        
        # 按时间排序以确保正确匹配
        compute_starts.sort(key=lambda x: x[0])
        compute_ends.sort(key=lambda x: x[0])
        
        # 对于可能存在的阶段匹配逻辑，我们按出现顺序配对：第 i 个 start 对应第 i 个 end（如果阶段相同则最好）
        min_count = min(len(compute_starts), len(compute_ends))
        if len(compute_starts) != len(compute_ends):
            print(f"警告: {server_name} Batch {batch_key} 的计算开始({len(compute_starts)})和结束({len(compute_ends)})事件数量不匹配，取最小值 {min_count} 进行匹配")
            compute_starts = compute_starts[:min_count]
            compute_ends = compute_ends[:min_count]
        
        for (start, batch_size, start_stage), (end, end_stage) in zip(compute_starts, compute_ends):
            # 优先使用 start_stage，如果为空使用 end_stage
            stage = start_stage or end_stage
            norm_start_ts = normalize(start)
            norm_end_ts = normalize(end)
            
            # 检查是否在时间范围内
            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                # 计算实际显示的起始点和结束点
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                
                # 计算持续时间（毫秒）
                duration_ms = (end - start) * 1000
                
                # 选择颜色（保持原色）
                color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                
                # 计算矩形高度和位置（用 Rectangle 来支持 hatch）
                bar_height = COMPUTE_BAR_HEIGHT
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
                
                # 根据阶段设置 hatch（不改变颜色）
                if stage:
                    stage_lower = stage.lower()
                    if 'prefill' in stage_lower:
                        rect.set_hatch('//')   # prefill 使用斜线
                    elif 'decode' in stage_lower:
                        rect.set_hatch('..')   # decode 使用点状
                    else:
                        # 其他阶段可以使用横线
                        rect.set_hatch('--')
                
                ax.add_patch(rect)
                
                # 添加图例句柄（只添加一次）
                if batch_id not in [h[0] for h in legend_handles if isinstance(h[0], int)]:
                    legend_handles.append((batch_id, plt.Line2D([0], [0], color=color, lw=4, alpha=ALPHA)))
                
                # 添加文本标注
                mid_x = (display_start + display_end) / 2
                text = f"bs={batch_size}\n{duration_ms:.1f}ms\n{stage or ''}"
                
                # 根据服务器位置决定文本标注位置
                if server_name == 'server1':
                    ann_y = resource_level - bar_height/2 - 0.15
                else:
                    ann_y = resource_level + bar_height/2 + 0.15
                
                ann = ax.text(
                    mid_x, ann_y, text,
                    ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                )
                annotations.append(ann)

# 绘制传输段函数 - 扩展支持4个服务器，使用完整的batch_key匹配
def draw_transfer_segments():
    global found_transfers
    
    print("\n绘制中间结果传输:")
    
    # S1 -> S2 传输
    print("绘制 S1->S2 传输...")
    # 获取所有可能的batch_key
    all_batch_keys = set(list(server1_events.keys()) + list(server2_events.keys()))
    
    for batch_key in all_batch_keys:
        # 获取server1的trans_start事件
        s1_trans_starts = [ts for etype, ts, _, _ in server1_events.get(batch_key, []) 
                          if etype == 'trans_start']
        s1_trans_starts.sort()
        
        # 获取server2的recv事件
        s2_recvs = [ts for etype, ts, _, _ in server2_events.get(batch_key, []) 
                   if etype == 'recv']
        s2_recvs.sort()
        
        # 确保有匹配的事件
        if s1_trans_starts and s2_recvs:
            for i in range(min(len(s1_trans_starts), len(s2_recvs))):
                start = s1_trans_starts[i]
                end = s2_recvs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                # 检查是否在时间范围内
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    duration_ms = (end - start) * 1000
                    # 从batch_key中提取batch_id用于颜色
                    try:
                        batch_id = int(batch_key.split()[0])
                    except:
                        batch_id = hash(batch_key) % len(BATCH_COLORS)
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    # 绘制传输段
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network S1→S2'] - bar_height / 2.0
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
                    
                    # 添加文本标注
                    mid_x = (display_start + display_end) / 2
                    text = f"S1→S2\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network S1→S2'] - bar_height/2 - 0.15
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("S1->S2", batch_key, start, end))
    
    # S2 -> S3 传输
    print("绘制 S2->S3 传输...")
    all_batch_keys = set(list(server2_events.keys()) + list(server3_events.keys()))
    
    for batch_key in all_batch_keys:
        s2_trans_starts = [ts for etype, ts, _, _ in server2_events.get(batch_key, []) 
                          if etype == 'trans_start']
        s2_trans_starts.sort()
        
        s3_recvs = [ts for etype, ts, _, _ in server3_events.get(batch_key, []) 
                   if etype == 'recv']
        s3_recvs.sort()
        
        if s2_trans_starts and s3_recvs:
            for i in range(min(len(s2_trans_starts), len(s3_recvs))):
                start = s2_trans_starts[i]
                end = s3_recvs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    duration_ms = (end - start) * 1000
                    try:
                        batch_id = int(batch_key.split()[0])
                    except:
                        batch_id = hash(batch_key) % len(BATCH_COLORS)
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network S2→S3'] - bar_height / 2.0
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
                    text = f"S2→S3\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network S2→S3'] - bar_height/2 - 0.15
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("S2->S3", batch_key, start, end))
    
    # S3 -> S4 传输
    print("绘制 S3->S4 传输...")
    all_batch_keys = set(list(server3_events.keys()) + list(server4_events.keys()))
    
    for batch_key in all_batch_keys:
        s3_trans_starts = [ts for etype, ts, _, _ in server3_events.get(batch_key, []) 
                          if etype == 'trans_start']
        s3_trans_starts.sort()
        
        s4_recvs = [ts for etype, ts, _, _ in server4_events.get(batch_key, []) 
                   if etype == 'recv']
        s4_recvs.sort()
        
        if s3_trans_starts and s4_recvs:
            for i in range(min(len(s3_trans_starts), len(s4_recvs))):
                start = s3_trans_starts[i]
                end = s4_recvs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    duration_ms = (end - start) * 1000
                    try:
                        batch_id = int(batch_key.split()[0])
                    except:
                        batch_id = hash(batch_key) % len(BATCH_COLORS)
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network S3→S4'] - bar_height / 2.0
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
                    text = f"S3→S4\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network S3→S4'] - bar_height/2 - 0.15
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("S3->S4", batch_key, start, end))
    
    # 返回传输 (所有服务器返回结果到S1)
    print("绘制返回传输...")
    # S4 -> S3 返回
    all_batch_keys = set(list(server4_events.keys()) + list(server3_events.keys()))
    
    for batch_key in all_batch_keys:
        s4_trans_starts = [ts for etype, ts, _, _ in server4_events.get(batch_key, []) 
                          if etype == 'trans_start']
        s4_trans_starts.sort()
        
        s3_backs = [ts for etype, ts, _, _ in server3_events.get(batch_key, []) 
                   if etype == 'back_to_head']
        s3_backs.sort()
        
        if s4_trans_starts and s3_backs:
            for i in range(min(len(s4_trans_starts), len(s3_backs))):
                start = s4_trans_starts[i]
                end = s3_backs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    duration_ms = (end - start) * 1000
                    try:
                        batch_id = int(batch_key.split()[0])
                    except:
                        batch_id = hash(batch_key) % len(BATCH_COLORS)
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network Returns'] - bar_height / 2.0
                    rect = Rectangle(
                        (display_start, y_bottom),
                        display_end - display_start,
                        bar_height,
                        facecolor=color,
                        edgecolor=color,
                        alpha=ALPHA * 0.7,  # 降低alpha以区分方向
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(rect)
                    
                    mid_x = (display_start + display_end) / 2
                    text = f"S4→S3\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Returns'] - bar_height/2 - 0.15
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("S4->S3", batch_key, start, end))
    
    # S3 -> S2 返回
    all_batch_keys = set(list(server3_events.keys()) + list(server2_events.keys()))
    
    for batch_key in all_batch_keys:
        s3_trans_starts = [ts for etype, ts, _, _ in server3_events.get(batch_key, []) 
                          if etype == 'trans_start']
        s3_trans_starts.sort()
        
        s2_backs = [ts for etype, ts, _, _ in server2_events.get(batch_key, []) 
                   if etype == 'back_to_head']
        s2_backs.sort()
        
        if s3_trans_starts and s2_backs:
            for i in range(min(len(s3_trans_starts), len(s2_backs))):
                start = s3_trans_starts[i]
                end = s2_backs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    duration_ms = (end - start) * 1000
                    try:
                        batch_id = int(batch_key.split()[0])
                    except:
                        batch_id = hash(batch_key) % len(BATCH_COLORS)
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network Returns'] - bar_height / 2.0
                    rect = Rectangle(
                        (display_start, y_bottom),
                        display_end - display_start,
                        bar_height,
                        facecolor=color,
                        edgecolor=color,
                        alpha=ALPHA * 0.7,
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(rect)
                    
                    mid_x = (display_start + display_end) / 2
                    text = f"S3→S2\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Returns'] - bar_height/2 - 0.15
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("S3->S2", batch_key, start, end))
    
    # S2 -> S1 返回
    all_batch_keys = set(list(server4_events.keys()) + list(server1_events.keys()))
    
    for batch_key in all_batch_keys:
        s4_trans_starts = [ts for etype, ts, _, _ in server4_events.get(batch_key, []) 
                          if etype == 'trans_start']
        s4_trans_starts.sort()
        
        s1_backs = [ts for etype, ts, _, _ in server1_events.get(batch_key, []) 
                   if etype == 'back_to_head']
        s1_backs.sort()
        
        if s4_trans_starts and s1_backs:
            for i in range(min(len(s4_trans_starts), len(s1_backs))):
                start = s4_trans_starts[i]
                end = s1_backs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    duration_ms = (end - start) * 1000
                    try:
                        batch_id = int(batch_key.split()[0])
                    except:
                        batch_id = hash(batch_key) % len(BATCH_COLORS)
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network Returns'] - bar_height / 2.0
                    rect = Rectangle(
                        (display_start, y_bottom),
                        display_end - display_start,
                        bar_height,
                        facecolor=color,
                        edgecolor=color,
                        alpha=ALPHA * 0.7,
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(rect)
                    
                    mid_x = (display_start + display_end) / 2
                    text = f"S4→S1\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Returns'] - bar_height/2 - 0.15
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("S4->S1", batch_key, start, end))

# 绘制所有部分
print("\n绘制Server1计算段...")
draw_compute_segments(server1_events, RESOURCE_LEVELS['Server1 Compute'], 'server1')
print("绘制Server2计算段...")
draw_compute_segments(server2_events, RESOURCE_LEVELS['Server2 Compute'], 'server2')
print("绘制Server3计算段...")
draw_compute_segments(server3_events, RESOURCE_LEVELS['Server3 Compute'], 'server3')
print("绘制Server4计算段...")
draw_compute_segments(server4_events, RESOURCE_LEVELS['Server4 Compute'], 'server4')
print("绘制网络传输段...")
draw_transfer_segments()
print("绘制请求事件...")
draw_request_events(server1_requests)

if not found_transfers:
    print("警告: 没有找到任何网络传输事件!")
else:
    print(f"成功绘制 {len(found_transfers)} 个传输事件")

# 设置图表属性
ax.set_yticks(list(RESOURCE_LEVELS.values()))
ax.set_yticklabels(list(RESOURCE_LEVELS.keys()), fontsize=11, fontweight='bold')
ax.set_ylabel('Resources', fontsize=12, fontweight='bold')
ax.set_xlabel(f'Time (seconds from base time {base_time:.6f})', fontsize=11)
ax.set_title(f'4-Server Distributed Computing Timeline', fontsize=14, fontweight='bold', pad=15)

# 设置x轴范围
ax.set_xlim(norm_start, norm_end)

# 设置y轴范围以包含所有层级
ax.set_ylim(-5.5, 9.0)

# 添加时间刻度线
ax.grid(True, axis='x', linestyle='--', alpha=0.6)

# 添加资源分隔线
for y in RESOURCE_LEVELS.values():
    ax.axhline(y=y, color='gray', alpha=0.3, linewidth=0.5)

# 创建专业图例
if legend_handles:
    # 分离batch和请求事件的图例
    batch_handles = [h[1] for h in legend_handles if isinstance(h[0], int)]
    batch_labels = [f'Batch {h[0]}' for h in legend_handles if isinstance(h[0], int)]
    
    request_handles = [h[1] for h in legend_handles if isinstance(h[0], str)]
    request_labels = [f'Request {h[0].title()}' for h in legend_handles if isinstance(h[0], str)]
    
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
    
    # 请求事件图例
    if request_handles:
        request_legend = plt.legend(
            request_handles,
            request_labels,
            title='Request Events',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.85),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(request_legend)

# 添加阶段样式的图例
stage_handles = [
    Rectangle((0, 0), 1, 1, facecolor='gray', alpha=ALPHA, hatch='//', label='Prefill'),
    Rectangle((0, 0), 1, 1, facecolor='gray', alpha=ALPHA, hatch='..', label='Decode'),
    Rectangle((0, 0), 1, 1, facecolor='gray', alpha=ALPHA, hatch='--', label='Other Stages')
]
stage_legend = plt.legend(
    handles=stage_handles,
    title='Compute Stages',
    loc='upper left',
    bbox_to_anchor=(0.01, 0.70),
    frameon=True,
    framealpha=0.9,
    edgecolor='#CCCCCC'
)
ax.add_artist(stage_legend)

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
    spine.set_linewidth(0.8)

# 调整标注位置避免重叠
def adjust_annotations(annotations, min_distance=0.02):
    """调整标注位置避免重叠"""
    # 按x坐标排序
    annotations.sort(key=lambda ann: ann.get_position()[0])
    
    # 检查并调整重叠
    for i in range(1, len(annotations)):
        prev = annotations[i-1]
        curr = annotations[i]
        prev_pos = prev.get_position()
        curr_pos = curr.get_position()
        
        # 检查x坐标是否太近
        if abs(curr_pos[0] - prev_pos[0]) < min_distance:
            # 垂直偏移调整
            offset = min_distance - abs(curr_pos[0] - prev_pos[0])
            new_y = curr_pos[1] - offset * 0.5
            curr.set_position((curr_pos[0], new_y))

# 应用标注调整
if annotations:
    adjust_annotations(annotations)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.95)

# 保存图像（可选）
plt.savefig('4server_distributed_timeline.pdf', bbox_inches='tight', dpi=300)
plt.show()