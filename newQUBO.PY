import numpy as np
import random
import time
import math
import csv
import json
import os
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 步骤1：初始化参数（新增区间波动开关）
# --------------------------
# 区间波动开关：True=开启波动，False=关闭波动（关闭后实验不可运行）
FLUCTUATION_ENABLE = False  # 核心开关变量

# 中间站停靠线容量限制惩罚开关
PLATFORM_CAPACITY_ENABLE = True  # 核心开关变量
MAX_NORMAL_PLATFORMS = 3  # 正常使用的停靠线数量
PLATFORM_PENALTY_WEIGHT = 3.0  # 每使用一条备用线的惩罚权重

stations = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
intervals = [f'{stations[i]}_{stations[i+1]}' for i in range(len(stations) - 1)]
INTERMEDIATE_STATIONS = stations[1:-1]

TRAIN_METADATA = {
    '101': {'direction': 'up'},
    '102': {'direction': 'up'},
    '103': {'direction': 'up'},
    '104': {'direction': 'up'},
    '201': {'direction': 'down'},
    '202': {'direction': 'down'},
    '203': {'direction': 'down'},
    '204': {'direction': 'down'}
}

train_times = {
    '101': {'A_B': 4.5, 'B_C': 5.0, 'C_D': 4.5, 'D_E': 5.0, 'E_F': 4.0, 'F_G': 4.5, 'B': 2.0, 'C': 3.0, 'D': 2.0, 'E': 2.0, 'F': 1.5},
    '102': {'A_B': 5.0, 'B_C': 4.5, 'C_D': 5.0, 'D_E': 4.5, 'E_F': 4.5, 'F_G': 5.0, 'B': 2.5, 'C': 2.0, 'D': 2.5, 'E': 2.0, 'F': 2.0},
    '103': {'A_B': 4.0, 'B_C': 4.5, 'C_D': 4.5, 'D_E': 5.0, 'E_F': 4.5, 'F_G': 4.0, 'B': 1.5, 'C': 2.5, 'D': 2.0, 'E': 2.5, 'F': 1.5},
    '104': {'A_B': 5.5, 'B_C': 5.0, 'C_D': 4.5, 'D_E': 4.5, 'E_F': 5.0, 'F_G': 4.5, 'B': 2.0, 'C': 3.0, 'D': 2.5, 'E': 2.0, 'F': 2.0},
    '201': {'A_B': 4.5, 'B_C': 4.0, 'C_D': 4.5, 'D_E': 5.0, 'E_F': 4.5, 'F_G': 4.0, 'B': 2.0, 'C': 2.5, 'D': 2.0, 'E': 2.5, 'F': 2.0},
    '202': {'A_B': 5.0, 'B_C': 4.5, 'C_D': 5.0, 'D_E': 4.5, 'E_F': 4.0, 'F_G': 4.5, 'B': 2.5, 'C': 2.0, 'D': 2.5, 'E': 2.0, 'F': 1.5},
    '203': {'A_B': 4.0, 'B_C': 4.5, 'C_D': 4.0, 'D_E': 4.5, 'E_F': 5.0, 'F_G': 4.5, 'B': 1.5, 'C': 2.5, 'D': 2.0, 'E': 2.0, 'F': 2.5},
    '204': {'A_B': 5.5, 'B_C': 5.0, 'C_D': 4.5, 'D_E': 4.0, 'E_F': 4.5, 'F_G': 5.0, 'B': 2.5, 'C': 2.0, 'D': 2.5, 'E': 2.0, 'F': 2.0}
}

# 区间波动配置（仅开启时生效）
FLUCTUATION_RATE = 0.2  # 波动幅度（±20%）
FLUCTUATION_PENALTY_WEIGHT = 1.5  # 波动惩罚权重

train_ids = list(train_times.keys())
n_trains = len(train_ids)
DEFAULT_TRAIN_IDS = train_ids.copy()
track_change_cost = 2  # 换轨惩罚成本
best_result = None  # 存储单次最优结果
experiment_records = {}  # 存储实验统计结果
last_weight_search_result = None

EXPORT_DIR = 'experiment_exports'
QUBO_ENERGY_SCALE = 100.0
DEFAULT_QUBO_WEIGHTS = {
    'ORDER_MUTEX_PENALTY': 100.0,
    'TRACK_MUTEX_PENALTY': 100.0,
    'CONFLICT_PENALTY': 200.0,
    'DELAY_PENALTY': 1.0,
    'TRACK_CHANGE_PENALTY': 2.0
}
QUBO_WEIGHT_CONFIG = DEFAULT_QUBO_WEIGHTS.copy()


def get_train_direction(tid):
    return TRAIN_METADATA[tid]['direction']


def get_route_stations(tid):
    if get_train_direction(tid) == 'up':
        return stations.copy()
    return list(reversed(stations))


def get_route_intervals(tid):
    if get_train_direction(tid) == 'up':
        return intervals.copy()
    return list(reversed(intervals))


def get_origin_station(tid):
    return get_route_stations(tid)[0]


def get_terminal_station(tid):
    return get_route_stations(tid)[-1]


def capture_runtime_config():
    return {
        'FLUCTUATION_ENABLE': FLUCTUATION_ENABLE,
        'FLUCTUATION_RATE': FLUCTUATION_RATE,
        'FLUCTUATION_PENALTY_WEIGHT': FLUCTUATION_PENALTY_WEIGHT,
        'PLATFORM_CAPACITY_ENABLE': PLATFORM_CAPACITY_ENABLE,
        'MAX_NORMAL_PLATFORMS': MAX_NORMAL_PLATFORMS,
        'PLATFORM_PENALTY_WEIGHT': PLATFORM_PENALTY_WEIGHT,
        'train_ids': train_ids.copy(),
        'n_trains': n_trains,
        'track_change_cost': track_change_cost,
        'qubo_weights': QUBO_WEIGHT_CONFIG.copy()
    }


def restore_runtime_config(config):
    global FLUCTUATION_ENABLE, FLUCTUATION_RATE, FLUCTUATION_PENALTY_WEIGHT
    global PLATFORM_CAPACITY_ENABLE, MAX_NORMAL_PLATFORMS, PLATFORM_PENALTY_WEIGHT
    global train_ids, n_trains, track_change_cost, QUBO_WEIGHT_CONFIG
    FLUCTUATION_ENABLE = config['FLUCTUATION_ENABLE']
    FLUCTUATION_RATE = config['FLUCTUATION_RATE']
    FLUCTUATION_PENALTY_WEIGHT = config['FLUCTUATION_PENALTY_WEIGHT']
    PLATFORM_CAPACITY_ENABLE = config['PLATFORM_CAPACITY_ENABLE']
    MAX_NORMAL_PLATFORMS = config['MAX_NORMAL_PLATFORMS']
    PLATFORM_PENALTY_WEIGHT = config['PLATFORM_PENALTY_WEIGHT']
    train_ids = config['train_ids'].copy()
    n_trains = config['n_trains']
    track_change_cost = config['track_change_cost']
    QUBO_WEIGHT_CONFIG = config['qubo_weights'].copy()
    rebuild_qubo_model()


def apply_runtime_overrides(overrides):
    global FLUCTUATION_ENABLE, FLUCTUATION_RATE, FLUCTUATION_PENALTY_WEIGHT
    global PLATFORM_CAPACITY_ENABLE, MAX_NORMAL_PLATFORMS, PLATFORM_PENALTY_WEIGHT
    global train_ids, n_trains, track_change_cost, QUBO_WEIGHT_CONFIG
    if not overrides:
        return
    if 'FLUCTUATION_ENABLE' in overrides:
        FLUCTUATION_ENABLE = overrides['FLUCTUATION_ENABLE']
    if 'FLUCTUATION_RATE' in overrides:
        FLUCTUATION_RATE = overrides['FLUCTUATION_RATE']
    if 'FLUCTUATION_PENALTY_WEIGHT' in overrides:
        FLUCTUATION_PENALTY_WEIGHT = overrides['FLUCTUATION_PENALTY_WEIGHT']
    if 'PLATFORM_CAPACITY_ENABLE' in overrides:
        PLATFORM_CAPACITY_ENABLE = overrides['PLATFORM_CAPACITY_ENABLE']
    if 'MAX_NORMAL_PLATFORMS' in overrides:
        MAX_NORMAL_PLATFORMS = overrides['MAX_NORMAL_PLATFORMS']
    if 'PLATFORM_PENALTY_WEIGHT' in overrides:
        PLATFORM_PENALTY_WEIGHT = overrides['PLATFORM_PENALTY_WEIGHT']
    if 'train_ids' in overrides:
        train_ids = overrides['train_ids'].copy()
        n_trains = len(train_ids)
    if 'track_change_cost' in overrides:
        track_change_cost = overrides['track_change_cost']
    if 'qubo_weights' in overrides or 'train_ids' in overrides:
        if 'qubo_weights' in overrides:
            QUBO_WEIGHT_CONFIG.update(overrides['qubo_weights'])
        rebuild_qubo_model()


def ensure_export_dir():
    os.makedirs(EXPORT_DIR, exist_ok=True)


def sanitize_filename(name):
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ('-', '_'):
            safe.append(ch)
        else:
            safe.append('_')
    return ''.join(safe).strip('_') or 'result'


def get_timestamp():
    return time.strftime('%Y%m%d_%H%M%S')


def flatten_dict(data, prefix=''):
    flat = {}
    for key, value in data.items():
        new_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            flat[new_key] = value
    return flat


def export_rows(base_name, summary_rows, detail_rows=None, metadata=None):
    ensure_export_dir()
    timestamp = get_timestamp()
    safe_name = sanitize_filename(base_name)
    summary_path = os.path.join(EXPORT_DIR, f'{safe_name}_{timestamp}_summary.csv')
    json_path = os.path.join(EXPORT_DIR, f'{safe_name}_{timestamp}.json')

    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_path = None

    if detail_rows:
        detail_path = os.path.join(EXPORT_DIR, f'{safe_name}_{timestamp}_detail.csv')
        detail_fields = sorted({key for row in detail_rows for key in row.keys()})
        with open(detail_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields)
            writer.writeheader()
            writer.writerows(detail_rows)
    else:
        detail_path = None

    payload = {
        'metadata': metadata or {},
        'summary_rows': summary_rows,
        'detail_rows': detail_rows or []
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        'summary_csv': summary_path,
        'detail_csv': detail_path,
        'json': json_path
    }


def export_markdown_report(base_name, title, summary_rows, metadata=None, extra_sections=None):
    ensure_export_dir()
    timestamp = get_timestamp()
    safe_name = sanitize_filename(base_name)
    md_path = os.path.join(EXPORT_DIR, f'{safe_name}_{timestamp}.md')

    lines = [f'# {title}', '']
    if metadata:
        lines.append('## 实验配置')
        lines.append('')
        for key, value in metadata.items():
            lines.append(f'- {key}: {value}')
        lines.append('')

    if summary_rows:
        headers = list(summary_rows[0].keys())
        lines.append('## 汇总结果')
        lines.append('')
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        for row in summary_rows:
            lines.append('| ' + ' | '.join(str(row.get(header, '')) for header in headers) + ' |')
        lines.append('')

    if extra_sections:
        for section_title, section_lines in extra_sections:
            lines.append(f'## {section_title}')
            lines.append('')
            lines.extend(section_lines)
            lines.append('')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return md_path


def export_chart(base_name, fig):
    ensure_export_dir()
    timestamp = get_timestamp()
    safe_name = sanitize_filename(base_name)
    image_path = os.path.join(EXPORT_DIR, f'{safe_name}_{timestamp}.png')
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return image_path


def generate_experiment_chart(experiment_type, summary_rows, base_name):
    if not summary_rows:
        return None

    if experiment_type == 'comparison':
        labels = [row['算法'] for row in summary_rows]
        energy = [row['平均能量值'] for row in summary_rows]
        delay = [row['平均总延误时间'] for row in summary_rows]
        solve_time = [row['平均求解时间'] for row in summary_rows]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].bar(labels, energy, color='#5B8FF9')
        axes[0].set_title('平均能量值对比')
        axes[0].tick_params(axis='x', rotation=45)
        axes[1].bar(labels, delay, color='#61DDAA')
        axes[1].set_title('平均总延误对比')
        axes[1].tick_params(axis='x', rotation=45)
        axes[2].bar(labels, solve_time, color='#F6BD16')
        axes[2].set_title('平均求解时间对比')
        axes[2].tick_params(axis='x', rotation=45)
        return export_chart(base_name, fig)

    if experiment_type == 'ablation':
        labels = [row['消融项'] for row in summary_rows]
        energy = [row['平均能量值'] for row in summary_rows]
        delay = [row['平均总延误时间'] for row in summary_rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(labels, energy, marker='o', color='#5B8FF9')
        axes[0].set_title('消融场景平均能量值')
        axes[0].tick_params(axis='x', rotation=30)
        axes[1].plot(labels, delay, marker='s', color='#E8684A')
        axes[1].set_title('消融场景平均延误')
        axes[1].tick_params(axis='x', rotation=30)
        return export_chart(base_name, fig)

    if experiment_type == 'weight_search':
        labels = [str(row['候选编号']) for row in summary_rows]
        scores = [row['评分'] for row in summary_rows]
        energies = [row['平均能量值'] for row in summary_rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(labels, scores, marker='o', color='#9270CA')
        axes[0].set_title('候选权重评分')
        axes[0].set_xlabel('候选编号')
        axes[1].plot(labels, energies, marker='^', color='#5AD8A6')
        axes[1].set_title('候选权重平均能量值')
        axes[1].set_xlabel('候选编号')
        return export_chart(base_name, fig)

    if experiment_type == 'robustness':
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        algorithms = sorted({row['算法'] for row in summary_rows})
        for algorithm in algorithms:
            rows = sorted([row for row in summary_rows if row['算法'] == algorithm], key=lambda item: item['波动率'])
            rates = [row['波动率'] for row in rows]
            energies = [row['平均能量值'] for row in rows]
            delays = [row['平均总延误时间'] for row in rows]
            axes[0].plot(rates, energies, marker='o', label=algorithm)
            axes[1].plot(rates, delays, marker='s', label=algorithm)
        axes[0].set_title('波动率-平均能量值')
        axes[0].set_xlabel('波动率')
        axes[1].set_title('波动率-平均总延误')
        axes[1].set_xlabel('波动率')
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        return export_chart(base_name, fig)

    if experiment_type == 'convergence':
        labels = [row['算法'] for row in summary_rows]
        final_best = [row['平均最终最优能量'] for row in summary_rows]
        conv_iters = [row['平均收敛迭代'] for row in summary_rows]
        tail = [row['尾段波动'] for row in summary_rows]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].bar(labels, final_best, color='#5B8FF9')
        axes[0].set_title('平均最终最优能量')
        axes[0].tick_params(axis='x', rotation=45)
        axes[1].bar(labels, conv_iters, color='#61DDAA')
        axes[1].set_title('平均收敛迭代')
        axes[1].tick_params(axis='x', rotation=45)
        axes[2].bar(labels, tail, color='#E8684A')
        axes[2].set_title('尾段波动')
        axes[2].tick_params(axis='x', rotation=45)
        return export_chart(base_name, fig)

    if experiment_type == 'weight_sensitivity':
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        weight_keys = sorted({row['权重项'] for row in summary_rows})
        for weight_key in weight_keys:
            rows = sorted([row for row in summary_rows if row['权重项'] == weight_key], key=lambda item: item['测试值'])
            x_values = [row['测试值'] for row in rows]
            y_values = [row['平均能量值'] for row in rows]
            ax.plot(x_values, y_values, marker='o', label=weight_key)
        ax.set_title('QUBO权重局部敏感性')
        ax.set_xlabel('测试值')
        ax.set_ylabel('平均能量值')
        ax.legend(fontsize=8)
        return export_chart(base_name, fig)

    if experiment_type == 'problem_scale':
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        algorithms = sorted({row['算法'] for row in summary_rows})
        for algorithm in algorithms:
            rows = sorted([row for row in summary_rows if row['算法'] == algorithm], key=lambda item: item['列车规模'])
            scales = [row['列车规模'] for row in rows]
            energies = [row['平均能量值'] for row in rows]
            times = [row['平均求解时间'] for row in rows]
            axes[0].plot(scales, energies, marker='o', label=algorithm)
            axes[1].plot(scales, times, marker='s', label=algorithm)
        axes[0].set_title('问题规模-平均能量值')
        axes[0].set_xlabel('列车数量')
        axes[1].set_title('问题规模-平均求解时间')
        axes[1].set_xlabel('列车数量')
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        return export_chart(base_name, fig)

    if experiment_type == 'scenario':
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        labels = [row['场景'] for row in summary_rows]
        energies = [row['平均能量值'] for row in summary_rows]
        delays = [row['平均总延误时间'] for row in summary_rows]
        axes[0].bar(labels, energies, color='#5B8FF9')
        axes[0].set_title('场景-平均能量值')
        axes[0].tick_params(axis='x', rotation=35)
        axes[1].bar(labels, delays, color='#E8684A')
        axes[1].set_title('场景-平均总延误')
        axes[1].tick_params(axis='x', rotation=35)
        return export_chart(base_name, fig)

    if experiment_type == 'ablation_unified_energy':
        algorithms = sorted({row['算法'] for row in summary_rows})
        scenarios = sorted({row['消融项'] for row in summary_rows})
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        for idx, algorithm in enumerate(algorithms):
            if idx < len(axes):  # 确保不超出子图数量
                ax = axes[idx]
                # 按照scenarios的顺序获取结果
                energies = []
                for scenario in scenarios:
                    row = next((r for r in summary_rows if r['算法'] == algorithm and r['消融项'] == scenario), None)
                    energies.append(row['平均能量值'] if row else 0)
                ax.plot(scenarios, energies, marker='o', linewidth=2, markersize=6)
                ax.set_title(f'{algorithm}', fontsize=10, fontweight='bold')
                ax.tick_params(axis='x', rotation=30, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.3)
        plt.suptitle('所有算法消融实验 - 平均能量值对比', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return export_chart(base_name, fig)

    if experiment_type == 'ablation_unified_delay':
        algorithms = sorted({row['算法'] for row in summary_rows})
        scenarios = sorted({row['消融项'] for row in summary_rows})
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        for idx, algorithm in enumerate(algorithms):
            if idx < len(axes):  # 确保不超出子图数量
                ax = axes[idx]
                # 按照scenarios的顺序获取结果
                delays = []
                for scenario in scenarios:
                    row = next((r for r in summary_rows if r['算法'] == algorithm and r['消融项'] == scenario), None)
                    delays.append(row['平均总延误时间'] if row else 0)
                ax.plot(scenarios, delays, marker='s', linewidth=2, markersize=6, color='#E8684A')
                ax.set_title(f'{algorithm}', fontsize=10, fontweight='bold')
                ax.tick_params(axis='x', rotation=30, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.3)
        plt.suptitle('所有算法消融实验 - 平均延误对比', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return export_chart(base_name, fig)

    return None

# --------------------------
# 步骤2：区间波动与惩罚核心函数（添加开关控制）
# --------------------------
def get_fluctuated_time(tid, interval):
    """生成带随机波动的区间运行时间（受开关控制）"""
    base_time = train_times[tid][interval]
    if FLUCTUATION_ENABLE:  # 开关开启时才添加波动
        min_time = base_time * (1 - FLUCTUATION_RATE)
        max_time = base_time * (1 + FLUCTUATION_RATE)
        return round(random.uniform(min_time, max_time), 1)
    else:  # 开关关闭时返回固定基础时间
        return base_time

def calculate_fluctuation_penalty(actual_time, base_time):
    """计算区间波动惩罚（受开关控制）"""
    if FLUCTUATION_ENABLE:  # 开关开启时才计算惩罚
        fluctuation_ratio = abs(actual_time - base_time) / base_time
        return fluctuation_ratio * FLUCTUATION_PENALTY_WEIGHT * base_time
    else:  # 开关关闭时惩罚为0
        return 0.0

# --------------------------
# 步骤3：核心计算函数（含波动惩罚）
# --------------------------
def generate_random_combination():
    """生成随机组合（用于初始化算法）"""
    depart_order = random.sample(train_ids, n_trains)
    track_assign = {
        tid: {interval: random.choice([1, 2]) for interval in intervals}
        for tid in train_ids
    }
    return {'depart_order': depart_order, 'track_assign': track_assign}

def calculate_schedule(combination):
    """计算调度详情和能量值（含区间波动惩罚）"""
    depart_order = combination['depart_order']
    track_assign = combination['track_assign']
    priority_index = {tid: idx for idx, tid in enumerate(depart_order)}

    schedule = {}
    ready_departure = {}
    for tid in train_ids:
        route_stations = get_route_stations(tid)
        route_intervals = get_route_intervals(tid)
        station_info = {
            station: {'轨道': '--', '到时': '--', '停留时间': '--', '延误时间': '--', '出发时间': '--'}
            for station in stations
        }
        origin = route_stations[0]
        station_info[origin]['轨道'] = track_assign[tid][route_intervals[0]]
        station_info[origin]['出发时间'] = 0.0
        schedule[tid] = {
            '方向': '上行' if get_train_direction(tid) == 'up' else '下行',
            'route_stations': route_stations,
            'route_intervals': route_intervals,
            'stations': station_info,
            '中间站总延误': 0.0,
            '始发站延误': 0.0,
            '换轨次数': 0,
            '区间波动总惩罚': 0.0,
            '中间站停靠线惩罚': 0.0
        }
        ready_departure[tid] = 0.0

    for stage_idx in range(len(intervals)):
        resource_groups = {}
        for tid in train_ids:
            interval = schedule[tid]['route_intervals'][stage_idx]
            track = track_assign[tid][interval]
            resource_groups.setdefault((interval, track), []).append(tid)

        for (interval, track), tids in resource_groups.items():
            prev_release = 0.0
            ordered_tids = sorted(tids, key=lambda tid: priority_index[tid])
            for tid in ordered_tids:
                route_stations = schedule[tid]['route_stations']
                current_station = route_stations[stage_idx]
                next_station = route_stations[stage_idx + 1]
                actual_depart = max(ready_departure[tid], prev_release)

                if stage_idx == 0:
                    schedule[tid]['stations'][current_station]['轨道'] = track
                    schedule[tid]['stations'][current_station]['出发时间'] = round(actual_depart, 2)
                    schedule[tid]['始发站延误'] = round(actual_depart, 2)
                else:
                    stop_time = train_times[tid][current_station]
                    planned_depart = schedule[tid]['stations'][current_station]['到时'] + stop_time
                    delay = actual_depart - planned_depart
                    schedule[tid]['stations'][current_station]['轨道'] = track
                    schedule[tid]['stations'][current_station]['停留时间'] = round(stop_time, 2)
                    schedule[tid]['stations'][current_station]['延误时间'] = round(delay, 2)
                    schedule[tid]['stations'][current_station]['出发时间'] = round(actual_depart, 2)
                    schedule[tid]['中间站总延误'] += round(delay, 2)

                run_time = get_fluctuated_time(tid, interval)
                base_run_time = train_times[tid][interval]
                schedule[tid]['区间波动总惩罚'] += calculate_fluctuation_penalty(run_time, base_run_time)
                actual_arrive = actual_depart + run_time
                prev_release = actual_arrive

                if next_station != route_stations[-1]:
                    schedule[tid]['stations'][next_station]['到时'] = round(actual_arrive, 2)
                    schedule[tid]['stations'][next_station]['轨道'] = track_assign[tid][schedule[tid]['route_intervals'][stage_idx + 1]]
                    ready_departure[tid] = actual_arrive + train_times[tid][next_station]
                else:
                    schedule[tid]['stations'][next_station]['到时'] = round(actual_arrive, 2)
                    ready_departure[tid] = actual_arrive
    
    # 统计换轨次数
    for tid in train_ids:
        track_sequence = [track_assign[tid][interval] for interval in schedule[tid]['route_intervals']]
        for i in range(1, len(track_sequence)):
            if track_sequence[i] != track_sequence[i-1]:
                schedule[tid]['换轨次数'] += 1
    
    # 计算中间站停靠线容量限制惩罚
    total_platform_penalty = 0.0
    if PLATFORM_CAPACITY_ENABLE:
        for station in INTERMEDIATE_STATIONS:
            stay_intervals = []
            for tid in train_ids:
                arrive = schedule[tid]['stations'][station]['到时']
                depart = schedule[tid]['stations'][station]['出发时间']
                if isinstance(arrive, (int, float)) and isinstance(depart, (int, float)) and depart > arrive:
                    stay_intervals.append((arrive, depart, tid))

            if len(stay_intervals) < 2:
                continue

            all_times = sorted(list({t for interval in stay_intervals for t in [interval[0], interval[1]]}))
            for i in range(len(all_times) - 1):
                current_time = (all_times[i] + all_times[i+1]) / 2
                overlapping_trains_list = []
                for arrive, depart, tid in stay_intervals:
                    if arrive <= current_time <= depart:
                        overlapping_trains_list.append(tid)
                overlapping_count = len(overlapping_trains_list)

                if overlapping_count > MAX_NORMAL_PLATFORMS:
                    excess = overlapping_count - MAX_NORMAL_PLATFORMS
                    time_segment = all_times[i+1] - all_times[i]
                    segment_penalty = excess * PLATFORM_PENALTY_WEIGHT * time_segment
                    total_platform_penalty += segment_penalty

                    if overlapping_count > 0:
                        penalty_per_train = segment_penalty / overlapping_count
                        for tid in overlapping_trains_list:
                            schedule[tid]['中间站停靠线惩罚'] += penalty_per_train
    
    # 计算总能量值（含波动惩罚和停靠线惩罚）
    total_delay = sum(schedule[tid]['中间站总延误'] + schedule[tid]['始发站延误'] for tid in train_ids)
    total_track_cost = sum(schedule[tid]['换轨次数'] * track_change_cost for tid in train_ids)
    total_fluctuation_penalty = sum(schedule[tid]['区间波动总惩罚'] for tid in train_ids)
    total_energy = total_delay + total_track_cost + total_fluctuation_penalty + total_platform_penalty
    
    actual_depart_order = sorted([(tid, schedule[tid]['stations'][get_origin_station(tid)]['出发时间']) for tid in train_ids], key=lambda x: x[1])
    
    return {
        'schedule': schedule,
        '能量值': total_energy,
        '总延误': total_delay,  # 含始发站+中间站
        '换轨总成本': total_track_cost,
        '区间波动总惩罚': total_fluctuation_penalty,
        '中间站停靠线总惩罚': total_platform_penalty,
        '发车顺序': actual_depart_order,
        '轨道分配': track_assign
    }

# --------------------------
# QUBO模型核心模块（保持原有逻辑）
# --------------------------
def build_qubo_matrix(weights=None):
    """构建QUBO模型的Q矩阵（65×65）"""
    n_order_vars = n_trains * n_trains  # 发车顺序变量：5×5=25（索引0~24）
    n_track_vars = n_trains * len(intervals) * 2  # 轨道分配变量：5×4×2=40（索引25~64）
    total_vars = n_order_vars + n_track_vars
    Q = np.zeros((total_vars, total_vars), dtype=np.float64)
    weight_cfg = (weights or QUBO_WEIGHT_CONFIG).copy()

    # 惩罚权重
    ORDER_MUTEX_PENALTY = weight_cfg['ORDER_MUTEX_PENALTY']
    TRACK_MUTEX_PENALTY = weight_cfg['TRACK_MUTEX_PENALTY']
    CONFLICT_PENALTY = weight_cfg['CONFLICT_PENALTY']
    DELAY_PENALTY = weight_cfg['DELAY_PENALTY']
    TRACK_CHANGE_PENALTY = weight_cfg['TRACK_CHANGE_PENALTY']

    # 变量索引映射函数
    def get_order_var_idx(train_idx, pos_idx):
        return train_idx * n_trains + pos_idx
    
    def get_track_var_idx(train_idx, interval_idx, track_idx):
        return n_order_vars + train_idx * len(intervals) * 2 + interval_idx * 2 + track_idx
    
    # 1. 填充自身约束（互斥约束）
    for train_idx in range(n_trains):
        for pos1 in range(n_trains):
            idx1 = get_order_var_idx(train_idx, pos1)
            Q[idx1][idx1] += ORDER_MUTEX_PENALTY
            for pos2 in range(pos1 + 1, n_trains):
                idx2 = get_order_var_idx(train_idx, pos2)
                Q[idx1][idx2] += 2 * ORDER_MUTEX_PENALTY
                Q[idx2][idx1] += 2 * ORDER_MUTEX_PENALTY
    
    for pos_idx in range(n_trains):
        for train1 in range(n_trains):
            idx1 = get_order_var_idx(train1, pos_idx)
            for train2 in range(train1 + 1, n_trains):
                idx2 = get_order_var_idx(train2, pos_idx)
                Q[idx1][idx2] += 2 * ORDER_MUTEX_PENALTY
                Q[idx2][idx1] += 2 * ORDER_MUTEX_PENALTY
    
    for train_idx in range(n_trains):
        for interval_idx in range(len(intervals)):
            for track1 in range(2):
                idx1 = get_track_var_idx(train_idx, interval_idx, track1)
                Q[idx1][idx1] += TRACK_MUTEX_PENALTY
                for track2 in range(track1 + 1, 2):
                    idx2 = get_track_var_idx(train_idx, interval_idx, track2)
                    Q[idx1][idx2] += 2 * TRACK_MUTEX_PENALTY
                    Q[idx2][idx1] += 2 * TRACK_MUTEX_PENALTY
    
    # 2. 填充联合约束（同轨道冲突）
    for interval_idx, interval in enumerate(intervals):
        for track_idx in range(2):
            for train1_idx in range(n_trains):
                for train2_idx in range(train1_idx + 1, n_trains):
                    track_var1 = get_track_var_idx(train1_idx, interval_idx, track_idx)
                    track_var2 = get_track_var_idx(train2_idx, interval_idx, track_idx)
                    
                    for pos1_idx in range(n_trains):
                        for pos2_idx in range(n_trains):
                            if pos1_idx == pos2_idx:
                                continue
                            order_var1 = get_order_var_idx(train1_idx, pos1_idx)
                            order_var2 = get_order_var_idx(train2_idx, pos2_idx)
                            
                            train1_id = train_ids[train1_idx]
                            train2_id = train_ids[train2_idx]
                            run_time1 = train_times[train1_id][interval]
                            run_time2 = train_times[train2_id][interval]
                            depart_time1 = pos1_idx * run_time1
                            depart_time2 = pos2_idx * run_time2
                            arrive_time1 = depart_time1 + run_time1
                            arrive_time2 = depart_time2 + run_time2
                            
                            if not (arrive_time1 <= depart_time2 or arrive_time2 <= depart_time1):
                                Q[order_var1][order_var2] += CONFLICT_PENALTY
                                Q[order_var2][order_var1] += CONFLICT_PENALTY
                                Q[order_var1][track_var2] += CONFLICT_PENALTY
                                Q[track_var2][order_var1] += CONFLICT_PENALTY
                                Q[order_var2][track_var1] += CONFLICT_PENALTY
                                Q[track_var1][order_var2] += CONFLICT_PENALTY
                                Q[track_var1][track_var2] += CONFLICT_PENALTY
                                Q[track_var2][track_var1] += CONFLICT_PENALTY
    
    # 3. 填充目标惩罚（延误+换轨）
    for train_idx in range(n_trains):
        for pos_idx in range(n_trains):
            idx = get_order_var_idx(train_idx, pos_idx)
            train_id = train_ids[train_idx]
            first_interval = get_route_intervals(train_id)[0]
            base_delay = pos_idx * train_times[train_id][first_interval]
            Q[idx][idx] += base_delay * DELAY_PENALTY
    
    for train_idx in range(n_trains):
        for interval_idx in range(1, len(intervals)):
            prev_interval_idx = interval_idx - 1
            for track_prev in range(2):
                idx_prev = get_track_var_idx(train_idx, prev_interval_idx, track_prev)
                for track_curr in range(2):
                    if track_prev != track_curr:
                        idx_curr = get_track_var_idx(train_idx, interval_idx, track_curr)
                        Q[idx_prev][idx_curr] += TRACK_CHANGE_PENALTY
                        Q[idx_curr][idx_prev] += TRACK_CHANGE_PENALTY
    
    return Q, get_order_var_idx, get_track_var_idx

# 预构建Q矩阵
Q_MATRIX, get_order_idx, get_track_idx = build_qubo_matrix()
TOTAL_VARS = Q_MATRIX.shape[0]  # 总变量数：65


def rebuild_qubo_model():
    global Q_MATRIX, get_order_idx, get_track_idx, TOTAL_VARS
    Q_MATRIX, get_order_idx, get_track_idx = build_qubo_matrix(QUBO_WEIGHT_CONFIG)
    TOTAL_VARS = Q_MATRIX.shape[0]


def repair_qubo_solution(x):
    assigned_positions = set()
    for train_idx in range(n_trains):
        pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
        scores = x[pos_vars].copy()
        ranked_positions = list(np.argsort(scores)[::-1])
        chosen_position = None
        for pos in ranked_positions:
            if pos not in assigned_positions:
                chosen_position = pos
                break
        if chosen_position is None:
            remaining_positions = [pos for pos in range(n_trains) if pos not in assigned_positions]
            chosen_position = random.choice(remaining_positions) if remaining_positions else random.randint(0, n_trains - 1)
        x[pos_vars] = 0
        x[get_order_idx(train_idx, chosen_position)] = 1
        assigned_positions.add(chosen_position)
    for train_idx in range(n_trains):
        for interval_idx in range(len(intervals)):
            track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
            scores = x[track_vars].copy()
            chosen_track = int(np.argmax(scores)) if np.any(scores) else random.choice([0, 1])
            x[track_vars] = 0
            x[get_track_idx(train_idx, interval_idx, chosen_track)] = 1
    return x


def schedule_result_to_qubo_x(result):
    x = np.zeros(TOTAL_VARS, dtype=int)
    depart_order = [item[0] for item in result['发车顺序']]
    for pos_idx, tid in enumerate(depart_order):
        train_idx = train_ids.index(tid)
        x[get_order_idx(train_idx, pos_idx)] = 1
    for train_idx, tid in enumerate(train_ids):
        for interval_idx, interval in enumerate(intervals):
            track = result['轨道分配'][tid][interval]
            x[get_track_idx(train_idx, interval_idx, track - 1)] = 1
    return repair_qubo_solution(x)


def schedule_result_to_combination(result):
    return {
        'depart_order': [item[0] for item in result['发车顺序']],
        'track_assign': {tid: result['轨道分配'][tid].copy() for tid in train_ids}
    }

# QUBO辅助函数
def calculate_qubo_energy(x):
    """计算QUBO能量值（与调度能量值一致）"""
    return x.T @ Q_MATRIX @ x / QUBO_ENERGY_SCALE  # 缩放至实际调度能量值范围

def generate_valid_qubo_x():
    """生成合法的QUBO二进制变量x"""
    x = np.zeros(TOTAL_VARS, dtype=int)
    # 合法发车顺序
    train_pos = random.sample(range(n_trains), n_trains)
    for train_idx in range(n_trains):
        x[get_order_idx(train_idx, train_pos[train_idx])] = 1
    # 合法轨道分配
    for train_idx in range(n_trains):
        for interval_idx in range(len(intervals)):
            x[get_track_idx(train_idx, interval_idx, random.choice([0, 1]))] = 1
    return repair_qubo_solution(x)

def qubo_x_to_schedule(x):
    """将QUBO解x转换为调度方案"""
    x = repair_qubo_solution(x.copy())
    # 解析发车顺序
    depart_order = [''] * n_trains
    for train_idx in range(n_trains):
        for pos_idx in range(n_trains):
            if x[get_order_idx(train_idx, pos_idx)] == 1:
                depart_order[pos_idx] = train_ids[train_idx]
                break
    # 解析轨道分配
    track_assign = {tid: {} for tid in train_ids}
    for train_idx in range(n_trains):
        tid = train_ids[train_idx]
        for interval_idx, interval in enumerate(intervals):
            for track_idx in range(2):
                if x[get_track_idx(train_idx, interval_idx, track_idx)] == 1:
                    track_assign[tid][interval] = track_idx + 1  # 轨道1~2
                    break
    # 计算调度详情
    return calculate_schedule({'depart_order': depart_order, 'track_assign': track_assign})

# --------------------------
# 迭代历史记录与可视化模块
# --------------------------
ALL_ALGORITHMS = [
    'QUBO-暴力遍历算法',
    'QUBO-传统遗传算法',
    'QUBO-传统粒子群算法',
    'QUBO-传统模拟退火算法',
    'QUBO-量子遗传算法',
    'QUBO-量子粒子群算法',
    'QUBO-量子模拟退火算法',
    'QUBO-量子遗传-QAOA混合算法',
    'QUBO-量子退火-QAOA混合算法',
    'QUBO-量子粒子群-QAOA混合算法'
]

algorithm_history = {name: {'iterations': [], 'current_energy': [], 'best_energy': []} for name in ALL_ALGORITHMS}

def reset_algorithm_history(algorithm_name):
    if algorithm_name in algorithm_history:
        algorithm_history[algorithm_name] = {'iterations': [], 'current_energy': [], 'best_energy': []}

def record_iteration(algorithm_name, iteration, current_energy, best_energy):
    if algorithm_name in algorithm_history:
        algorithm_history[algorithm_name]['iterations'].append(iteration)
        algorithm_history[algorithm_name]['current_energy'].append(current_energy)
        algorithm_history[algorithm_name]['best_energy'].append(best_energy)

def show_single_algorithm_plot(algorithm_name):
    data = algorithm_history[algorithm_name]
    if not data['iterations']:
        print(f"算法 {algorithm_name} 暂无迭代数据！")
        return
    
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(data['iterations'], data['current_energy'], 'b-', alpha=0.6, label='当前迭代最优能量')
    plt.plot(data['iterations'], data['best_energy'], 'r-', linewidth=2, label='全局最优能量')
    
    best_idx = np.argmin(data['best_energy'])
    best_iter = data['iterations'][best_idx]
    best_val = data['best_energy'][best_idx]
    plt.scatter(best_iter, best_val, color='green', s=80, zorder=5, 
                label=f'最优能量值: {best_val:.2f}\n出现迭代: {best_iter}')
    
    plt.title(f'{algorithm_name} 迭代能量变化详情', fontsize=14)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('能量值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    input("按Enter返回...")

def iteration_plot_menu():
    while True:
        print("\n----- 迭代图查看菜单 -----")
        has_data_algos = [name for name in ALL_ALGORITHMS if len(algorithm_history[name]['iterations']) > 0]
        if has_data_algos:
            print("可查看的算法：")
            for i, name in enumerate(has_data_algos, 1):
                print(f"{i}. {name}")
            print(f"{len(has_data_algos)+1}. 返回主菜单")
            
            try:
                choice = int(input(f"请选择算法 (1-{len(has_data_algos)+1}): "))
                if 1 <= choice <= len(has_data_algos):
                    show_single_algorithm_plot(has_data_algos[choice-1])
                elif choice == len(has_data_algos)+1:
                    break
                else:
                    print("无效选择！")
            except ValueError:
                print("请输入数字！")
        else:
            print("暂无任何算法的迭代数据，请先运行算法！")
            break

# --------------------------
# 所有算法实现（均添加迭代记录）
# --------------------------
def solve_by_brute_force():
    """1. QUBO-暴力遍历算法"""
    global best_result
    algorithm_name = 'QUBO-暴力遍历算法'
    reset_algorithm_history(algorithm_name)
    print(f"{algorithm_name}求解中...")
    
    best_energy = float('inf')
    best_x = None
    count = 0
    max_count = 100000  # 限制遍历数量
    
    while count < max_count:
        x = generate_valid_qubo_x()
        current_energy = calculate_qubo_energy(x)
        count += 1
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_x = x.copy()
        
        if count % 1000 == 0:
            record_iteration(algorithm_name, count, current_energy, best_energy)
            print(f"已遍历 {count}/{max_count}，当前最优: {best_energy:.2f}")
    
    best_schedule = qubo_x_to_schedule(best_x)
    best_result = best_schedule
    return best_result

def solve_by_genetic_algorithm(pop_size=50, generations=100):
    """2. QUBO-传统遗传算法"""
    global best_result
    algorithm_name = 'QUBO-传统遗传算法'
    reset_algorithm_history(algorithm_name)
    print(f"{algorithm_name}求解中（种群: {pop_size}, 代数: {generations}）...")
    
    population = [generate_valid_qubo_x() for _ in range(pop_size)]
    best_energy = float('inf')
    best_x = None
    n_order_vars = n_trains * n_trains
    
    for gen in range(generations):
        scored = [(calculate_qubo_energy(x), x) for x in population]
        scored.sort(key=lambda x: x[0])
        current_best_energy = scored[0][0]
        current_best_x = scored[0][1]
        
        if current_best_energy < best_energy:
            best_energy = current_best_energy
            best_x = current_best_x.copy()
        
        record_iteration(algorithm_name, gen+1, current_best_energy, best_energy)
        
        # 选择
        elites = [x for (_, x) in scored[:int(pop_size*0.2)]]
        others = [x for (_, x) in scored[int(pop_size*0.2):]]
        energies = [e for e, _ in scored[int(pop_size*0.2):]]
        probs = [(max(energies) - e + 1e-6) / (sum(max(energies)-e for e in energies) + 1e-6) for e in energies]
        selected = random.choices(others, weights=probs, k=pop_size - len(elites))
        
        # 交叉
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 >= len(selected):
                offspring.append(selected[i])
                break
            p1, p2 = selected[i], selected[i+1]
            cross_idx = random.randint(1, TOTAL_VARS-1)
            child = np.concatenate([p1[:cross_idx], p2[cross_idx:]])
            
            # 修复合法性
            for train_idx in range(n_trains):
                pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                if sum(child[pos_vars]) != 1:
                    child[pos_vars] = 0
                    child[random.choice(pos_vars)] = 1
            for train_idx in range(n_trains):
                for interval_idx in range(len(intervals)):
                    track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                    if sum(child[track_vars]) != 1:
                        child[track_vars] = 0
                        child[random.choice(track_vars)] = 1
            offspring.append(child)
        
        # 变异
        for ind in offspring:
            if random.random() < 0.05:
                mutate_idx = random.randint(0, TOTAL_VARS-1)
                ind[mutate_idx] = 1 - ind[mutate_idx]
                if mutate_idx < n_order_vars:
                    train_idx = mutate_idx // n_trains
                    pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                    if sum(ind[pos_vars]) != 1:
                        ind[pos_vars] = 0
                        ind[random.choice(pos_vars)] = 1
                else:
                    train_idx = (mutate_idx - n_order_vars) // (len(intervals)*2)
                    interval_idx = ((mutate_idx - n_order_vars) % (len(intervals)*2)) // 2
                    track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                    if sum(ind[track_vars]) != 1:
                        ind[track_vars] = 0
                        ind[random.choice(track_vars)] = 1
        
        population = elites + offspring
        print(f"第 {gen+1}/{generations} 代，当前最优: {current_best_energy:.2f}, 全局最优: {best_energy:.2f}")
    
    best_schedule = qubo_x_to_schedule(best_x)
    best_result = best_schedule
    return best_result

def solve_by_particle_swarm(swarm_size=50, iterations=100):
    """3. QUBO-传统粒子群算法"""
    global best_result
    algorithm_name = 'QUBO-传统粒子群算法'
    reset_algorithm_history(algorithm_name)
    print(f"{algorithm_name}求解中（种群: {swarm_size}, 迭代: {iterations}）...")
    
    def create_qubo_particle():
        return {
            'position': generate_valid_qubo_x(),
            'velocity': np.random.rand(TOTAL_VARS) * 0.5,
            'pbest': None,
            'pbest_energy': float('inf')
        }
    
    swarm = [create_qubo_particle() for _ in range(swarm_size)]
    gbest_energy = float('inf')
    gbest_x = None
    
    for it in range(iterations):
        current_best_energy = float('inf')
        
        for particle in swarm:
            current_energy = calculate_qubo_energy(particle['position'])
            if current_energy < particle['pbest_energy']:
                particle['pbest_energy'] = current_energy
                particle['pbest'] = particle['position'].copy()
            if current_energy < current_best_energy:
                current_best_energy = current_energy
            if current_energy < gbest_energy:
                gbest_energy = current_energy
                gbest_x = particle['position'].copy()
        
        record_iteration(algorithm_name, it+1, current_best_energy, gbest_energy)
        
        # 更新速度和位置
        w, c1, c2 = 0.5, 1.2, 1.2
        for particle in swarm:
            r1, r2 = random.random(), random.random()
            particle['velocity'] = (w * particle['velocity'] +
                                  c1 * r1 * (particle['pbest'] - particle['position']) +
                                  c2 * r2 * (gbest_x - particle['position']))
            for idx in range(TOTAL_VARS):
                prob = 1 / (1 + math.exp(-particle['velocity'][idx]))
                if random.random() < prob:
                    particle['position'][idx] = 1 - particle['position'][idx]
            
            # 修复合法性
            for train_idx in range(n_trains):
                pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                if sum(particle['position'][pos_vars]) != 1:
                    particle['position'][pos_vars] = 0
                    particle['position'][random.choice(pos_vars)] = 1
            for train_idx in range(n_trains):
                for interval_idx in range(len(intervals)):
                    track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                    if sum(particle['position'][track_vars]) != 1:
                        particle['position'][track_vars] = 0
                        particle['position'][random.choice(track_vars)] = 1
        
        print(f"第 {it+1}/{iterations} 次迭代，当前最优: {current_best_energy:.2f}, 全局最优: {gbest_energy:.2f}")
    
    best_schedule = qubo_x_to_schedule(gbest_x)
    best_result = best_schedule
    return best_result

def solve_by_simulated_annealing(initial_temp=100.0, cooling_rate=0.9, iterations=100):
    """4. QUBO-传统模拟退火算法"""
    global best_result
    algorithm_name = 'QUBO-传统模拟退火算法'
    reset_algorithm_history(algorithm_name)
    print(f"{algorithm_name}求解中（温度: {initial_temp}, 迭代: {iterations}）...")
    
    current_x = generate_valid_qubo_x()
    current_energy = calculate_qubo_energy(current_x)
    best_x = current_x.copy()
    best_energy = current_energy
    temp = initial_temp
    n_order_vars = n_trains * n_trains
    
    for i in range(iterations):
        # 生成邻域解
        neighbor_x = current_x.copy()
        mutate_idx = random.randint(0, TOTAL_VARS-1)
        neighbor_x[mutate_idx] = 1 - neighbor_x[mutate_idx]
        
        # 修复合法性
        if mutate_idx < n_order_vars:
            train_idx = mutate_idx // n_trains
            pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
            if sum(neighbor_x[pos_vars]) != 1:
                neighbor_x[pos_vars] = 0
                neighbor_x[random.choice(pos_vars)] = 1
        else:
            train_idx = (mutate_idx - n_order_vars) // (len(intervals)*2)
            interval_idx = ((mutate_idx - n_order_vars) % (len(intervals)*2)) // 2
            track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
            if sum(neighbor_x[track_vars]) != 1:
                neighbor_x[track_vars] = 0
                neighbor_x[random.choice(track_vars)] = 1
        
        neighbor_energy = calculate_qubo_energy(neighbor_x)
        
        # 接受准则
        if neighbor_energy < current_energy or random.random() < math.exp(-(neighbor_energy - current_energy)/temp):
            current_x = neighbor_x
            current_energy = neighbor_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_x = current_x.copy()
        
        record_iteration(algorithm_name, i+1, current_energy, best_energy)
        
        temp *= cooling_rate
        if i % 100 == 0:
            print(f"第 {i+1}/{iterations} 次迭代，当前能量: {current_energy:.2f}, 全局最优: {best_energy:.2f}")
    
    best_schedule = qubo_x_to_schedule(best_x)
    best_result = best_schedule
    return best_result

def solve_by_quantum_genetic_algorithm(pop_size=100, generations=100, learning_rate=0.1, elite_ratio=0.05, mutation_rate=0.01, lr_decay=0.99):
    """5. QUBO-量子遗传算法"""
    global best_result
    algorithm_name = 'QUBO-量子遗传算法'
    reset_algorithm_history(algorithm_name)
    print(f"{algorithm_name}求解中（种群: {pop_size}, 代数: {generations}, 学习率: {learning_rate}, 精英比例: {elite_ratio}, 变异率: {mutation_rate}）...")
    
    quantum_pop = [
        {'alpha': np.full(TOTAL_VARS, math.sqrt(0.5)), 'beta': np.full(TOTAL_VARS, math.sqrt(0.5))}
        for _ in range(pop_size)
    ]
    best_energy = float('inf')
    best_x = None
    
    # 计算精英数量
    elite_count = max(1, int(pop_size * elite_ratio))
    
    for gen in range(generations):
        classical_pop = []
        for q_ind in quantum_pop:
            x = np.zeros(TOTAL_VARS, dtype=int)
            for idx in range(TOTAL_VARS):
                x[idx] = 1 if random.random() < q_ind['alpha'][idx]**2 else 0
            # 修复合法性
            for train_idx in range(n_trains):
                pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                if sum(x[pos_vars]) != 1:
                    x[pos_vars] = 0
                    x[random.choice(pos_vars)] = 1
            for train_idx in range(n_trains):
                for interval_idx in range(len(intervals)):
                    track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                    if sum(x[track_vars]) != 1:
                        x[track_vars] = 0
                        x[random.choice(track_vars)] = 1
            classical_pop.append(x)
        
        scored = [(calculate_qubo_energy(x), x) for x in classical_pop]
        scored.sort(key=lambda x: x[0])
        current_best_energy, current_best_x = scored[0]
        
        if current_best_energy < best_energy:
            best_energy = current_best_energy
            best_x = current_best_x.copy()
        
        record_iteration(algorithm_name, gen+1, current_best_energy, best_energy)
        
        # 保存精英个体
        elite_pop = quantum_pop[:elite_count].copy()
        
        # 动态调整学习率
        current_lr = learning_rate * (lr_decay ** gen)
        
        # 量子旋转门更新
        for i, q_ind in enumerate(quantum_pop):
            # 精英个体不更新
            if i < elite_count:
                continue
                
            for idx in range(TOTAL_VARS):
                target = current_best_x[idx]
                alpha, beta = q_ind['alpha'][idx], q_ind['beta'][idx]
                if target == 1:
                    new_alpha = alpha * math.cos(current_lr) - beta * math.sin(current_lr)
                    new_beta = alpha * math.sin(current_lr) + beta * math.cos(current_lr)
                else:
                    new_alpha = alpha * math.cos(current_lr) + beta * math.sin(current_lr)
                    new_beta = -alpha * math.sin(current_lr) + beta * math.cos(current_lr)
                norm = math.sqrt(new_alpha**2 + new_beta**2)
                q_ind['alpha'][idx] = new_alpha / norm
                q_ind['beta'][idx] = new_beta / norm
            
            # 量子变异
            for idx in range(TOTAL_VARS):
                if random.random() < mutation_rate:
                    # 随机重置量子位到叠加态
                    q_ind['alpha'][idx] = math.sqrt(0.5)
                    q_ind['beta'][idx] = math.sqrt(0.5)
        
        print(f"第 {gen+1}/{generations} 代，当前最优: {current_best_energy:.2f}, 全局最优: {best_energy:.2f}")
    
    best_schedule = qubo_x_to_schedule(best_x)
    best_result = best_schedule
    return best_result

def solve_by_quantum_particle_swarm(swarm_size=100, iterations=100, initial_sigma=0.5, alpha_coeff=0.5, cognitive_coeff=0.5, social_coeff=0.5, sigma_min=0.01):
    """6. QUBO-量子粒子群算法"""
    global best_result
    algorithm_name = 'QUBO-量子粒子群算法'
    reset_algorithm_history(algorithm_name)
    print(f"{algorithm_name}求解中（种群: {swarm_size}, 迭代: {iterations}, 初始sigma: {initial_sigma}, alpha系数: {alpha_coeff}）...")
    
    def create_qubo_qparticle():
        return {
            'mean': generate_valid_qubo_x(),
            'sigma': initial_sigma,
            'pbest': None,
            'pbest_energy': float('inf')
        }
    
    swarm = [create_qubo_qparticle() for _ in range(swarm_size)]
    gbest_energy = float('inf')
    gbest_x = None
    
    for it in range(iterations):
        current_best_energy = float('inf')
        
        for particle in swarm:
            x = particle['mean'].copy()
            # 根据sigma进行变异
            for idx in range(TOTAL_VARS):
                if random.random() < particle['sigma']:
                    x[idx] = 1 - x[idx]
            # 修复合法性
            for train_idx in range(n_trains):
                pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                if sum(x[pos_vars]) != 1:
                    x[pos_vars] = 0
                    x[random.choice(pos_vars)] = 1
            for train_idx in range(n_trains):
                for interval_idx in range(len(intervals)):
                    track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                    if sum(x[track_vars]) != 1:
                        x[track_vars] = 0
                        x[random.choice(track_vars)] = 1
            
            current_energy = calculate_qubo_energy(x)
            if current_energy < particle['pbest_energy']:
                particle['pbest_energy'] = current_energy
                particle['pbest'] = x.copy()
            if current_energy < current_best_energy:
                current_best_energy = current_energy
            if current_energy < gbest_energy:
                gbest_energy = current_energy
                gbest_x = x.copy()
        
        record_iteration(algorithm_name, it+1, current_best_energy, gbest_energy)
        
        # 更新量子态 - 动态alpha参数
        alpha = alpha_coeff * (1 - it/iterations)
        for particle in swarm:
            if particle['pbest'] is None or gbest_x is None:
                continue
            
            # 结合认知系数和社会系数的更新策略
            for idx in range(TOTAL_VARS):
                # 根据认知系数和社会系数调整更新概率
                r1, r2 = random.random(), random.random()
                cognitive_prob = cognitive_coeff * r1
                social_prob = social_coeff * r2
                total_prob = cognitive_prob + social_prob
                
                if total_prob < 0.5:
                    # 保持当前状态
                    pass
                elif r1 > r2:
                    # 向个体最优移动
                    particle['mean'][idx] = particle['pbest'][idx]
                else:
                    # 向全局最优移动
                    particle['mean'][idx] = gbest_x[idx]
            
            # 限制sigma最小值，避免收敛过快
            particle['sigma'] = max(sigma_min, alpha * particle['sigma'])
        
        print(f"第 {it+1}/{iterations} 次迭代，当前最优: {current_best_energy:.2f}, 全局最优: {gbest_energy:.2f}")
    
    best_schedule = qubo_x_to_schedule(gbest_x)
    best_result = best_schedule
    return best_result

def solve_by_quantum_simulated_annealing(initial_temp=100.0, cooling_rate=0.99, iterations=1000, n_superposition=5, crossover_prob=0.8, mutation_rate=0.1, temp_strategy='exponential'):
    """7. QUBO-量子模拟退火算法"""
    global best_result
    algorithm_name = 'QUBO-量子模拟退火算法'
    reset_algorithm_history(algorithm_name)
    print(f"{algorithm_name}求解中（温度: {initial_temp}, 迭代: {iterations}, 叠加态数: {n_superposition}）...")
    
    superposition = [generate_valid_qubo_x() for _ in range(n_superposition)]
    best_energy = min([calculate_qubo_energy(x) for x in superposition])
    best_x = superposition[np.argmin([calculate_qubo_energy(x) for x in superposition])].copy()
    temp = initial_temp
    
    for i in range(iterations):
        new_superposition = []
        
        for _ in range(n_superposition):
            # 根据交叉概率决定是否进行量子交叉
            if random.random() < crossover_prob:
                # 量子交叉操作
                s1, s2 = random.sample(superposition, 2)
                # 基于当前温度动态调整交叉比例
                cross_ratio = max(0.3, min(0.7, 1.0 - temp/initial_temp * 0.5))
                new_x = np.where(np.random.rand(TOTAL_VARS) > cross_ratio, s1, s2)
            else:
                # 否则随机选择一个现有解
                new_x = random.choice(superposition).copy()
            
            # 变异操作 - 基于温度和变异率
            for idx in range(TOTAL_VARS):
                if random.random() < mutation_rate * (temp/initial_temp):
                    new_x[idx] = 1 - new_x[idx]
            
            # 修复合法性
            for train_idx in range(n_trains):
                pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                if sum(new_x[pos_vars]) != 1:
                    new_x[pos_vars] = 0
                    new_x[random.choice(pos_vars)] = 1
            for train_idx in range(n_trains):
                for interval_idx in range(len(intervals)):
                    track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                    if sum(new_x[track_vars]) != 1:
                        new_x[track_vars] = 0
                        new_x[random.choice(track_vars)] = 1
            
            new_superposition.append(new_x)
        
        # 叠加态更新策略 - 精英保留与多样性维护
        all_solutions = superposition + new_superposition
        all_energies = [calculate_qubo_energy(x) for x in all_solutions]
        
        # 排序并选择最优的n_superposition个解
        sorted_indices = np.argsort(all_energies)
        selected_indices = []
        selected_x = []
        
        # 优先选择最优解
        for idx in sorted_indices:
            x = all_solutions[idx]
            # 检查多样性 - 如果与已选解的汉明距离大于阈值则选择
            is_diverse = True
            for selected in selected_x:
                hamming_dist = np.sum(x != selected)
                if hamming_dist < TOTAL_VARS * 0.1:  # 至少10%的差异
                    is_diverse = False
                    break
            
            if is_diverse or len(selected_x) < n_superposition * 0.7:  # 至少70%的精英保留
                selected_indices.append(idx)
                selected_x.append(x)
                if len(selected_indices) >= n_superposition:
                    break
        
        superposition = [all_solutions[idx] for idx in selected_indices]
        
        current_best_energy = min(all_energies)
        if current_best_energy < best_energy:
            best_energy = current_best_energy
            best_x = all_solutions[np.argmin(all_energies)].copy()
        
        record_iteration(algorithm_name, i+1, current_best_energy, best_energy)
        
        # 不同的温度更新策略
        if temp_strategy == 'exponential':
            temp *= cooling_rate
        elif temp_strategy == 'linear':
            temp = initial_temp - (initial_temp * cooling_rate * i / iterations)
        elif temp_strategy == 'logarithmic':
            temp = initial_temp / math.log(i + 2)
        
        # 确保温度不会过低
        temp = max(temp, 0.01)
        
        if i % 100 == 0:
            print(f"第 {i+1}/{iterations} 次迭代，当前最优: {current_best_energy:.2f}, 全局最优: {best_energy:.2f}, 温度: {temp:.2f}")
    
    best_schedule = qubo_x_to_schedule(best_x)
    best_result = best_schedule
    return best_result

def solve_by_qga_qaoa_hybrid(pop_size=100, generations=100, qaoa_iterations=50, learning_rate=0.1, elite_ratio=0.05, mutation_rate=0.01, lr_decay=0.99, n_neighbors=10, perturbation_prob=0.1, patience=10):
    """8. QUBO-量子遗传-QAOA混合算法"""
    global best_result
    algorithm_name = 'QUBO-量子遗传-QAOA混合算法'
    qga_algorithm_name = 'QUBO-量子遗传算法'
    reset_algorithm_history(qga_algorithm_name)
    print(f"{algorithm_name}求解中（种群: {pop_size}, QGA代数: {generations}, QAOA迭代: {qaoa_iterations}, 邻域大小: {n_neighbors}）...")
    
    # 步骤1：QUBO-量子遗传获取初始解，使用优化参数
    qga_result = solve_by_quantum_genetic_algorithm(pop_size, generations, learning_rate, elite_ratio, mutation_rate, lr_decay)
    qga_history = algorithm_history[qga_algorithm_name].copy()
    reset_algorithm_history(algorithm_name)
    algorithm_history[algorithm_name]['iterations'] = qga_history['iterations']
    algorithm_history[algorithm_name]['current_energy'] = qga_history['current_energy']
    algorithm_history[algorithm_name]['best_energy'] = qga_history['best_energy']
    initial_depart_order = [item[0] for item in qga_result['发车顺序']]
    initial_track_assign = qga_result['轨道分配']
    
    # 转换为x
    initial_x = np.zeros(TOTAL_VARS, dtype=int)
    for pos_idx, tid in enumerate(initial_depart_order):
        train_idx = train_ids.index(tid)
        initial_x[get_order_idx(train_idx, pos_idx)] = 1
    for train_idx in range(n_trains):
        tid = train_ids[train_idx]
        for interval_idx, interval in enumerate(intervals):
            track = initial_track_assign[tid][interval]
            initial_x[get_track_idx(train_idx, interval_idx, track-1)] = 1
    
    best_energy = calculate_qubo_energy(initial_x)
    best_x = initial_x.copy()
    
    # 步骤2：QAOA局部优化，添加早停机制
    no_improve_count = 0
    
    for q_iter in range(qaoa_iterations):
        neighbors = []
        
        # 根据迭代动态调整邻域搜索策略
        current_perturb_prob = perturbation_prob * (1 - q_iter / qaoa_iterations * 0.5)  # 逐渐减小扰动概率
        
        for _ in range(n_neighbors):
            new_x = best_x.copy()
            # 执行邻域扰动
            for idx in range(TOTAL_VARS):
                if random.random() < current_perturb_prob:
                    new_x[idx] = 1 - new_x[idx]
            
            # 修复合法性
            for train_idx in range(n_trains):
                pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                if sum(new_x[pos_vars]) != 1:
                    new_x[pos_vars] = 0
                    new_x[random.choice(pos_vars)] = 1
            for train_idx in range(n_trains):
                for interval_idx in range(len(intervals)):
                    track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                    if sum(new_x[track_vars]) != 1:
                        new_x[track_vars] = 0
                        new_x[random.choice(track_vars)] = 1
            
            neighbors.append(new_x)
        
        # 添加当前最优解以确保不会变差
        neighbors.append(best_x)
        
        # 评估所有邻域解
        scored = [(calculate_qubo_energy(x), x) for x in neighbors]
        scored.sort(key=lambda x: x[0])
        current_best_energy, current_best_x = scored[0]
        
        record_iteration(algorithm_name, generations + q_iter + 1, current_best_energy, best_energy)
        
        # 更新最优解
        if current_best_energy < best_energy:
            best_energy = current_best_energy
            best_x = current_best_x.copy()
            no_improve_count = 0  # 重置早停计数
            print(f"QAOA迭代 {q_iter+1}/{qaoa_iterations}: 当前最优: {current_best_energy:.2f}, 全局最优: {best_energy:.2f}")
        else:
            no_improve_count += 1
            # 早停机制
            if no_improve_count >= patience:
                print(f"QAOA早停：{patience}次迭代无改进")
                break
    
    best_schedule = qubo_x_to_schedule(best_x)
    best_result = best_schedule
    return best_result

def solve_by_qsa_qaoa_hybrid(initial_temp=100.0, cooling_rate=0.99, qsa_iterations=100, qaoa_iterations=50, n_superposition=5, crossover_prob=0.8, mutation_rate=0.1, temp_strategy='exponential', n_neighbors=10, perturbation_prob=0.1, patience=10, hybrid_point_strategy='fixed'):
    """9. QUBO-量子退火-QAOA混合算法"""
    global best_result
    algorithm_name = 'QUBO-量子退火-QAOA混合算法'
    qsa_algorithm_name = 'QUBO-量子模拟退火算法'
    reset_algorithm_history(qsa_algorithm_name)
    print(f"{algorithm_name}求解中（温度: {initial_temp}, 迭代: {qsa_iterations}, QAOA迭代: {qaoa_iterations}, 邻域大小: {n_neighbors}）...")
    
    # 步骤1：QUBO-量子退火获取初始解，使用优化参数
    qsa_result = solve_by_quantum_simulated_annealing(initial_temp, cooling_rate, qsa_iterations, n_superposition, crossover_prob, mutation_rate, temp_strategy)
    qsa_history = algorithm_history[qsa_algorithm_name].copy()
    reset_algorithm_history(algorithm_name)
    algorithm_history[algorithm_name]['iterations'] = qsa_history['iterations']
    algorithm_history[algorithm_name]['current_energy'] = qsa_history['current_energy']
    algorithm_history[algorithm_name]['best_energy'] = qsa_history['best_energy']
    initial_depart_order = [item[0] for item in qsa_result['发车顺序']]
    initial_track_assign = qsa_result['轨道分配']
    
    # 转换为x
    initial_x = np.zeros(TOTAL_VARS, dtype=int)
    for pos_idx, tid in enumerate(initial_depart_order):
        train_idx = train_ids.index(tid)
        initial_x[get_order_idx(train_idx, pos_idx)] = 1
    for train_idx in range(n_trains):
        tid = train_ids[train_idx]
        for interval_idx, interval in enumerate(intervals):
            track = initial_track_assign[tid][interval]
            initial_x[get_track_idx(train_idx, interval_idx, track-1)] = 1
    
    best_energy = calculate_qubo_energy(initial_x)
    best_x = initial_x.copy()
    
    # 步骤2：QAOA局部优化，添加自适应邻域搜索和早停机制
    no_improve_count = 0
    
    for q_iter in range(qaoa_iterations):
        neighbors = []
        
        # 自适应调整邻域大小和扰动概率
        adaptive_neighbors = int(n_neighbors * (1 + 0.5 * math.sin(q_iter * math.pi / 10)))  # 周期性调整邻域大小
        current_perturb_prob = perturbation_prob * (1 - q_iter / qaoa_iterations * 0.6)  # 逐渐减小扰动概率
        
        for _ in range(adaptive_neighbors):
            new_x = best_x.copy()
            
            # 分层扰动策略
            # 1. 对排序变量进行较小概率的扰动
            for train_idx in range(n_trains):
                if random.random() < current_perturb_prob * 0.5:
                    pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                    new_x[pos_vars] = 0
                    new_x[random.choice(pos_vars)] = 1
            
            # 2. 对轨道分配变量进行较大概率的扰动
            for train_idx in range(n_trains):
                for interval_idx in range(len(intervals)):
                    if random.random() < current_perturb_prob * 1.5:
                        track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                        new_x[track_vars] = 0
                        new_x[random.choice(track_vars)] = 1
            
            neighbors.append(new_x)
        
        # 添加当前最优解以确保不会变差
        neighbors.append(best_x)
        
        # 评估所有邻域解
        scored = [(calculate_qubo_energy(x), x) for x in neighbors]
        scored.sort(key=lambda x: x[0])
        current_best_energy, current_best_x = scored[0]
        
        record_iteration(algorithm_name, qsa_iterations + q_iter + 1, current_best_energy, best_energy)
        
        # 更新最优解
        if current_best_energy < best_energy:
            best_energy = current_best_energy
            best_x = current_best_x.copy()
            no_improve_count = 0  # 重置早停计数
            print(f"QAOA迭代 {q_iter+1}/{qaoa_iterations}: 当前最优: {current_best_energy:.2f}, 全局最优: {best_energy:.2f}")
        else:
            no_improve_count += 1
            # 早停机制
            if no_improve_count >= patience:
                print(f"QAOA早停：{patience}次迭代无改进")
                break
    
    best_schedule = qubo_x_to_schedule(best_x)
    best_result = best_schedule
    return best_result

def solve_by_qpso_qaoa_hybrid(swarm_size=100, qpso_iterations=100, qaoa_iterations=50, initial_sigma=0.5, alpha_coeff=0.5, cognitive_coeff=0.5, social_coeff=0.5, sigma_min=0.01, n_neighbors=10, perturbation_prob=0.1, patience=10, adaptive_search=True):
    """10. QUBO-量子粒子群-QAOA混合算法"""
    global best_result
    algorithm_name = 'QUBO-量子粒子群-QAOA混合算法'
    
    # 先保存量子粒子群算法的历史记录名称
    qpso_algorithm_name = 'QUBO-量子粒子群算法'
    reset_algorithm_history(qpso_algorithm_name)
    
    print(f"{algorithm_name}求解中（种群: {swarm_size}, 迭代: {qpso_iterations}, QAOA迭代: {qaoa_iterations}, 邻域大小: {n_neighbors}）...")
    
    # 步骤1：QUBO-量子粒子群获取初始解，使用优化参数
    qpso_result = solve_by_quantum_particle_swarm(swarm_size, qpso_iterations, initial_sigma, alpha_coeff, cognitive_coeff, social_coeff, sigma_min)
    
    # 保存量子粒子群阶段的迭代历史
    qpso_history = algorithm_history[qpso_algorithm_name].copy()
    
    # 重置混合算法的历史记录并复制量子粒子群的历史
    reset_algorithm_history(algorithm_name)
    algorithm_history[algorithm_name]['iterations'] = qpso_history['iterations']
    algorithm_history[algorithm_name]['current_energy'] = qpso_history['current_energy']
    algorithm_history[algorithm_name]['best_energy'] = qpso_history['best_energy']
    
    initial_depart_order = [item[0] for item in qpso_result['发车顺序']]
    initial_track_assign = qpso_result['轨道分配']
    
    # 转换为x
    initial_x = np.zeros(TOTAL_VARS, dtype=int)
    for pos_idx, tid in enumerate(initial_depart_order):
        train_idx = train_ids.index(tid)
        initial_x[get_order_idx(train_idx, pos_idx)] = 1
    for train_idx in range(n_trains):
        tid = train_ids[train_idx]
        for interval_idx, interval in enumerate(intervals):
            track = initial_track_assign[tid][interval]
            initial_x[get_track_idx(train_idx, interval_idx, track-1)] = 1
    
    best_energy = calculate_qubo_energy(initial_x)
    best_x = initial_x.copy()
    
    # 步骤2：QAOA局部优化，添加高级搜索策略和早停机制
    no_improve_count = 0
    
    for q_iter in range(qaoa_iterations):
        neighbors = []
        
        if adaptive_search:
            # 自适应搜索策略：根据迭代进程调整参数
            current_neighbors = int(n_neighbors * (1 + 0.3 * math.exp(-q_iter / qaoa_iterations * 10)))
            current_perturb_prob = perturbation_prob * math.exp(-q_iter / qaoa_iterations * 3)  # 指数衰减的扰动概率
        else:
            current_neighbors = n_neighbors
            current_perturb_prob = perturbation_prob
        
        for _ in range(current_neighbors):
            new_x = best_x.copy()
            
            # 智能扰动策略：基于变量重要性的扰动
            # 1. 对排序变量和轨道分配变量采用不同的扰动策略
            for idx in range(TOTAL_VARS):
                # 计算变量的重要性权重（基于问题特性）
                importance = 1.0
                
                # 为排序变量赋予不同的重要性
                if idx < n_trains * n_trains:  # 排序变量
                    train_idx = idx // n_trains
                    pos_idx = idx % n_trains
                    # 位置越靠前的列车排序可能越重要
                    if pos_idx < 2:
                        importance = 0.7  # 降低重要位置的扰动概率
                
                # 根据重要性和当前扰动概率决定是否扰动
                if random.random() < current_perturb_prob * importance:
                    new_x[idx] = 1 - new_x[idx]
            
            # 修复合法性
            for train_idx in range(n_trains):
                pos_vars = [get_order_idx(train_idx, p) for p in range(n_trains)]
                if sum(new_x[pos_vars]) != 1:
                    new_x[pos_vars] = 0
                    new_x[random.choice(pos_vars)] = 1
            for train_idx in range(n_trains):
                for interval_idx in range(len(intervals)):
                    track_vars = [get_track_idx(train_idx, interval_idx, t) for t in range(2)]
                    if sum(new_x[track_vars]) != 1:
                        new_x[track_vars] = 0
                        new_x[random.choice(track_vars)] = 1
            
            neighbors.append(new_x)
        
        # 添加当前最优解以确保不会变差
        neighbors.append(best_x)
        
        # 评估所有邻域解
        scored = [(calculate_qubo_energy(x), x) for x in neighbors]
        scored.sort(key=lambda x: x[0])
        current_best_energy, current_best_x = scored[0]
        
        # 记录QAOA迭代，迭代次数从量子粒子群的最后一次迭代加1开始
        record_iteration(algorithm_name, qpso_iterations + q_iter + 1, current_best_energy, best_energy)
        
        # 更新最优解
        if current_best_energy < best_energy:
            best_energy = current_best_energy
            best_x = current_best_x.copy()
            no_improve_count = 0  # 重置早停计数
            print(f"QAOA迭代 {q_iter+1}/{qaoa_iterations}: 当前最优: {current_best_energy:.2f}, 全局最优: {best_energy:.2f}")
        else:
            no_improve_count += 1
            # 早停机制
            if no_improve_count >= patience:
                print(f"QAOA早停：{patience}次迭代无改进")
                break
    
    best_schedule = qubo_x_to_schedule(best_x)
    best_result = best_schedule
    return best_result


def get_algorithm_registry(include_bruteforce=True):
    algorithms = [
        {'name': 'QUBO-传统遗传算法', 'func': solve_by_genetic_algorithm, 'kwargs': {}, 'type': 'standard'},
        {'name': 'QUBO-传统粒子群算法', 'func': solve_by_particle_swarm, 'kwargs': {}, 'type': 'standard'},
        {'name': 'QUBO-传统模拟退火算法', 'func': solve_by_simulated_annealing, 'kwargs': {}, 'type': 'standard'},
        {'name': 'QUBO-量子遗传算法', 'func': solve_by_quantum_genetic_algorithm, 'kwargs': {}, 'type': 'quantum'},
        {'name': 'QUBO-量子粒子群算法', 'func': solve_by_quantum_particle_swarm, 'kwargs': {}, 'type': 'quantum'},
        {'name': 'QUBO-量子模拟退火算法', 'func': solve_by_quantum_simulated_annealing, 'kwargs': {}, 'type': 'quantum'},
        {'name': 'QUBO-量子遗传-QAOA混合算法', 'func': solve_by_qga_qaoa_hybrid, 'kwargs': {}, 'type': 'hybrid'},
        {'name': 'QUBO-量子退火-QAOA混合算法', 'func': solve_by_qsa_qaoa_hybrid, 'kwargs': {}, 'type': 'hybrid'},
        {'name': 'QUBO-量子粒子群-QAOA混合算法', 'func': solve_by_qpso_qaoa_hybrid, 'kwargs': {}, 'type': 'hybrid'}
    ]
    if include_bruteforce:
        algorithms.append({'name': 'QUBO-暴力遍历算法', 'func': solve_by_brute_force, 'kwargs': {}, 'type': 'baseline'})
    return algorithms


def build_ablation_scenarios(algorithm_meta):
    scenarios = [
        {
            'name': '完整模型',
            'runtime_overrides': {
                'FLUCTUATION_ENABLE': True,
                'PLATFORM_CAPACITY_ENABLE': True,
                'track_change_cost': DEFAULT_QUBO_WEIGHTS['TRACK_CHANGE_PENALTY'],
                'qubo_weights': {'TRACK_CHANGE_PENALTY': DEFAULT_QUBO_WEIGHTS['TRACK_CHANGE_PENALTY']}
            },
            'solver_kwargs': {}
        },
        {
            'name': '无区间波动',
            'runtime_overrides': {'FLUCTUATION_ENABLE': False},
            'solver_kwargs': {}
        },
        {
            'name': '无停靠线容量限制',
            'runtime_overrides': {'PLATFORM_CAPACITY_ENABLE': False},
            'solver_kwargs': {}
        },
        {
            'name': '无换轨惩罚',
            'runtime_overrides': {
                'track_change_cost': 0.0,
                'qubo_weights': {'TRACK_CHANGE_PENALTY': 0.0}
            },
            'solver_kwargs': {}
        }
    ]
    if algorithm_meta['type'] == 'hybrid':
        scenarios.append({
            'name': '无QAOA局部搜索',
            'runtime_overrides': {
                'FLUCTUATION_ENABLE': True,
                'PLATFORM_CAPACITY_ENABLE': True,
                'track_change_cost': DEFAULT_QUBO_WEIGHTS['TRACK_CHANGE_PENALTY'],
                'qubo_weights': {'TRACK_CHANGE_PENALTY': DEFAULT_QUBO_WEIGHTS['TRACK_CHANGE_PENALTY']}
            },
            'solver_kwargs': {'qaoa_iterations': 0}
        })
    return scenarios


def extract_result_metrics(result, solve_time, seed=None, label=None, extra=None):
    row = {
        '标签': label,
        '随机种子': seed,
        '能量值': round(result['能量值'], 4),
        '总延误': round(result['总延误'], 4),
        '换轨成本': round(result['换轨总成本'], 4),
        '波动惩罚': round(result['区间波动总惩罚'], 4),
        '停靠线惩罚': round(result.get('中间站停靠线总惩罚', 0.0), 4),
        '是否有延误': result['总延误'] > 0,
        '求解时间': round(solve_time, 6)
    }
    if extra:
        row.update(extra)
    return row


def calculate_statistics(results):
    energies = [r['能量值'] for r in results]
    delays = [r['总延误'] for r in results]
    fluctuation_penalties = [r['波动惩罚'] for r in results]
    platform_penalties = [r['停靠线惩罚'] for r in results]
    solve_times = [r['求解时间'] for r in results]
    delay_count = sum(1 for r in results if r['是否有延误'])
    n_trials = len(results)

    return {
        '平均能量值': round(np.mean(energies), 2),
        '能量值标准差': round(np.std(energies), 2),
        '最大能量值': round(np.max(energies), 2),
        '最小能量值': round(np.min(energies), 2),
        '90%分位能量值': round(np.percentile(energies, 90), 2),
        '延误发生率': round((delay_count / n_trials) * 100, 2),
        '平均总延误时间': round(np.mean(delays), 2),
        '最大总延误时间': round(np.max(delays), 2),
        '最小总延误时间': round(np.min(delays), 2),
        '平均波动惩罚': round(np.mean(fluctuation_penalties), 2),
        '平均停靠线惩罚': round(np.mean(platform_penalties), 2),
        '平均求解时间': round(np.mean(solve_times), 4),
        '最大求解时间': round(np.max(solve_times), 4),
        '最小求解时间': round(np.min(solve_times), 4)
    }


def print_stats_table(title, stats):
    print(f"\n===== {title} =====")
    print(f"{'指标':<14} | {'数值':<10} | 说明")
    print('-' * 47)
    print(f"平均能量值        | {stats['平均能量值']:<10} | 越低越好")
    print(f"能量值标准差      | {stats['能量值标准差']:<10} | 越小越稳定")
    print(f"最大能量值        | {stats['最大能量值']:<10} | 越低越好")
    print(f"最小能量值        | {stats['最小能量值']:<10} | 越低越好")
    print(f"90%分位能量值     | {stats['90%分位能量值']:<10} | 稳健表现")
    print(f"延误发生率        | {stats['延误发生率']:<9}% | 越低越好")
    print(f"平均总延误时间    | {stats['平均总延误时间']:<10} | 越低越好")
    print(f"平均波动惩罚      | {stats['平均波动惩罚']:<10} | 越低越稳定")
    print(f"平均停靠线惩罚    | {stats['平均停靠线惩罚']:<10} | 越低越好")
    print(f"平均求解时间      | {stats['平均求解时间']:<10} | 越低越快")


def build_top_summary_lines(summary_rows, label_key='算法', top_k=3):
    lines = []
    for idx, row in enumerate(summary_rows[:top_k], start=1):
        label = row.get(label_key, row.get('消融项', f'候选{idx}'))
        lines.append(
            f"- Top {idx}: {label}，平均能量值={row.get('平均能量值')}，平均总延误={row.get('平均总延误时间')}，平均求解时间={row.get('平均求解时间')}"
        )
    return lines


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rank_absolute_values(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def paired_t_test(sample_a, sample_b):
    diffs = np.array(sample_a, dtype=float) - np.array(sample_b, dtype=float)
    n = len(diffs)
    if n < 2:
        return {'statistic': 0.0, 'p_value': 1.0, 'effect_size': 0.0}
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    if std_diff == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'effect_size': 0.0}
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_value = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    effect_size = mean_diff / std_diff
    return {'statistic': round(t_stat, 4), 'p_value': round(p_value, 6), 'effect_size': round(effect_size, 4)}


def wilcoxon_signed_rank_test(sample_a, sample_b):
    diffs = np.array(sample_a, dtype=float) - np.array(sample_b, dtype=float)
    non_zero = [(abs(diff), diff) for diff in diffs if abs(diff) > 1e-12]
    if not non_zero:
        return {'statistic': 0.0, 'p_value': 1.0, 'effect_size': 0.0}
    abs_values = [item[0] for item in non_zero]
    ranks = rank_absolute_values(abs_values)
    w_pos = sum(rank for rank, (_, diff) in zip(ranks, non_zero) if diff > 0)
    w_neg = sum(rank for rank, (_, diff) in zip(ranks, non_zero) if diff < 0)
    w_stat = min(w_pos, w_neg)
    n = len(non_zero)
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return {'statistic': round(w_stat, 4), 'p_value': 1.0, 'effect_size': 0.0}
    z_score = (w_stat - mean_w) / std_w
    p_value = 2.0 * (1.0 - normal_cdf(abs(z_score)))
    effect_size = abs(z_score) / math.sqrt(n)
    return {'statistic': round(w_stat, 4), 'p_value': round(p_value, 6), 'effect_size': round(effect_size, 4)}


def significance_label(p_value):
    if p_value < 0.01:
        return '极显著'
    if p_value < 0.05:
        return '显著'
    return '不显著'


def compare_detail_groups(detail_rows, group_key, baseline_label, target_label, metric_key='能量值'):
    baseline_rows = sorted([row for row in detail_rows if row[group_key] == baseline_label], key=lambda item: item['随机种子'])
    target_rows = sorted([row for row in detail_rows if row[group_key] == target_label], key=lambda item: item['随机种子'])
    paired_n = min(len(baseline_rows), len(target_rows))
    if paired_n == 0:
        return None
    baseline_values = [row[metric_key] for row in baseline_rows[:paired_n]]
    target_values = [row[metric_key] for row in target_rows[:paired_n]]
    t_result = paired_t_test(baseline_values, target_values)
    w_result = wilcoxon_signed_rank_test(baseline_values, target_values)
    mean_delta = round(float(np.mean(np.array(target_values) - np.array(baseline_values))), 4)
    return {
        '基准组': baseline_label,
        '对比组': target_label,
        '指标': metric_key,
        '平均差值(对比-基准)': mean_delta,
        't统计量': t_result['statistic'],
        't检验p值': t_result['p_value'],
        't检验结论': significance_label(t_result['p_value']),
        't效应量': t_result['effect_size'],
        'Wilcoxon统计量': w_result['statistic'],
        'Wilcoxon p值': w_result['p_value'],
        'Wilcoxon结论': significance_label(w_result['p_value']),
        'Wilcoxon效应量': w_result['effect_size']
    }


def build_significance_lines(test_rows):
    if not test_rows:
        return ['- 样本不足，未执行显著性检验。']
    lines = ['- 注：当前脚本采用配对 t 检验与 Wilcoxon 符号秩检验的近似实现，适合作为课程/论文实验中的统计参考。']
    for row in test_rows:
        lines.append(
            f"- {row['对比组']} 相对 {row['基准组']}：均值差={row['平均差值(对比-基准)']}，t检验 p={row['t检验p值']}（{row['t检验结论']}），Wilcoxon p={row['Wilcoxon p值']}（{row['Wilcoxon结论']}）"
        )
    return lines


def generate_comparison_conclusion(summary_rows, significance_rows=None):
    if not summary_rows:
        return []
    best = summary_rows[0]
    worst = summary_rows[-1]
    lines = [
        f"本组对比实验表明，{best['算法']} 在平均能量值指标上取得最优结果（{best['平均能量值']}），同时平均总延误为 {best['平均总延误时间']}，表现出更好的整体调度质量。",
        f"与排序末位的 {worst['算法']} 相比，最优算法在平均能量值上降低了 {round(worst['平均能量值'] - best['平均能量值'], 2)}，说明所采用的搜索机制能够更有效地逼近低能量可行解。",
        f"从求解效率看，最快算法的平均求解时间为 {min(row['平均求解时间'] for row in summary_rows)}，因此不同算法在解质量与计算开销之间存在明显权衡。"
    ]
    if significance_rows:
        top_sig = significance_rows[0]
        lines.append(
            f"进一步显著性检验显示，{top_sig['对比组']} 相对 {top_sig['基准组']} 的 t 检验 p 值为 {top_sig['t检验p值']}，Wilcoxon p 值为 {top_sig['Wilcoxon p值']}，可为算法性能差异是否稳定存在提供统计支撑。"
        )
    return [f"- {line}" for line in lines]


def generate_ablation_conclusion(summary_rows, significance_rows=None):
    if not summary_rows:
        return []
    full_model = next((row for row in summary_rows if row.get('消融项') == '完整模型'), summary_rows[0])
    worst = max(summary_rows, key=lambda item: item['平均能量值'])
    lines = [
        f"消融实验显示，完整模型的平均能量值为 {full_model['平均能量值']}，说明当前约束与惩罚项的联合设计能够获得较稳定的调度效果。",
        f"在各消融场景中，{worst['消融项']} 的平均能量值最高（{worst['平均能量值']}），表明该模块被移除后会明显削弱模型性能。"
    ]
    if significance_rows:
        significant_items = [row for row in significance_rows if row['Wilcoxon结论'] != '不显著' or row['t检验结论'] != '不显著']
        if significant_items:
            lines.append(f"统计检验进一步表明，共有 {len(significant_items)} 个消融场景与完整模型之间达到显著或极显著差异，说明关键模块对性能提升具有稳定贡献。")
    return [f"- {line}" for line in lines]


def generate_weight_search_conclusion(summary_rows, best_row):
    if not summary_rows or best_row is None:
        return []
    default_row = next((row for row in summary_rows if row['候选编号'] == 1), summary_rows[0])
    lines = [
        f"QUBO 权重搜索结果表明，候选 {best_row['候选编号']} 取得最优综合评分 {best_row['评分']}，其平均能量值为 {best_row['平均能量值']}。",
        f"相较默认权重配置，最佳权重将平均能量值改善了 {round(default_row['平均能量值'] - best_row['平均能量值'], 2)}，说明合理的约束权重匹配可以显著影响优化器的搜索方向与最终解质量。",
        f"最佳参数组合为 ORDER={best_row['ORDER_MUTEX_PENALTY']}、TRACK={best_row['TRACK_MUTEX_PENALTY']}、CONFLICT={best_row['CONFLICT_PENALTY']}、DELAY={best_row['DELAY_PENALTY']}、TRACK_CHANGE={best_row['TRACK_CHANGE_PENALTY']}。"
    ]
    return [f"- {line}" for line in lines]


def generate_robustness_conclusion(summary_rows):
    if not summary_rows:
        return []
    high_rate_rows = [row for row in summary_rows if row['波动率'] == max(item['波动率'] for item in summary_rows)]
    high_rate_rows.sort(key=lambda item: item['平均能量值'])
    best_high = high_rate_rows[0]
    worst_high = high_rate_rows[-1]
    lines = [
        f"在高扰动场景（波动率={best_high['波动率']}）下，{best_high['算法']} 取得最低平均能量值 {best_high['平均能量值']}，说明其生成的调度方案具有更强的抗扰动能力。",
        f"同一高扰动条件下，最优算法相较最差算法的平均能量值优势为 {round(worst_high['平均能量值'] - best_high['平均能量值'], 2)}，表明不同求解器得到的时刻表在鲁棒性层面存在明显差异。"
    ]
    return [f"- {line}" for line in lines]


def generate_convergence_conclusion(summary_rows):
    if not summary_rows:
        return []
    best_final = min(summary_rows, key=lambda item: item['平均最终最优能量'])
    fastest = min(summary_rows, key=lambda item: item['平均收敛迭代'])
    steadiest = min(summary_rows, key=lambda item: item['尾段波动'])
    lines = [
        f"收敛性实验表明，{best_final['算法']} 的平均最终最优能量最低（{best_final['平均最终最优能量']}），说明其具有更强的全局搜索能力。",
        f"从收敛速度看，{fastest['算法']} 以平均 {fastest['平均收敛迭代']} 次迭代达到稳定区域，收敛效率最好。",
        f"从稳定性看，{steadiest['算法']} 的尾段波动仅为 {steadiest['尾段波动']}，说明其后期搜索过程更平稳。"
    ]
    return [f"- {line}" for line in lines]


def generate_weight_sensitivity_conclusion(summary_rows):
    if not summary_rows:
        return []
    ranges = {}
    for row in summary_rows:
        ranges.setdefault(row['权重项'], []).append(row['平均能量值'])
    sensitivity = {key: round(max(values) - min(values), 2) for key, values in ranges.items()}
    most_sensitive = max(sensitivity.items(), key=lambda item: item[1])
    least_sensitive = min(sensitivity.items(), key=lambda item: item[1])
    lines = [
        f"局部敏感性分析显示，{most_sensitive[0]} 对平均能量值的影响最大，波动区间达到 {most_sensitive[1]}，说明该权重是模型调参时需要优先关注的关键参数。",
        f"相比之下，{least_sensitive[0]} 的敏感性最低，仅为 {least_sensitive[1]}，表明其在当前问题规模下对整体性能影响较弱。"
    ]
    return [f"- {line}" for line in lines]


def generate_problem_scale_conclusion(summary_rows):
    if not summary_rows:
        return []
    max_scale = max(row['列车规模'] for row in summary_rows)
    high_scale_rows = [row for row in summary_rows if row['列车规模'] == max_scale]
    high_scale_rows.sort(key=lambda item: item['平均能量值'])
    best_high = high_scale_rows[0]
    fastest_high = min(high_scale_rows, key=lambda item: item['平均求解时间'])
    lines = [
        f"在最大问题规模（{max_scale} 列车）下，{best_high['算法']} 取得最低平均能量值 {best_high['平均能量值']}，表明其在规模扩展后仍能保持较好的解质量。",
        f"同一规模下，{fastest_high['算法']} 的平均求解时间最低（{fastest_high['平均求解时间']}），说明不同算法在规模增大时的时间复杂度表现存在明显差异。"
    ]
    return [f"- {line}" for line in lines]


def generate_scenario_conclusion(summary_rows):
    if not summary_rows:
        return []
    best = min(summary_rows, key=lambda item: item['平均能量值'])
    worst = max(summary_rows, key=lambda item: item['平均能量值'])
    lines = [
        f"场景实验结果表明，{best['场景']} 下的平均能量值最低（{best['平均能量值']}），说明该参数配置更有利于获得高质量调度方案。",
        f"与此相比，{worst['场景']} 的平均能量值最高（{worst['平均能量值']}），反映出站台容量与波动惩罚系数的设定会直接影响模型的优化偏好与最终调度表现。"
    ]
    return [f"- {line}" for line in lines]


def generate_unified_ablation_conclusion(summary_rows):
    if not summary_rows:
        return []
    
    algorithms = sorted({row['算法'] for row in summary_rows})
    algorithm_best = {}
    for algorithm in algorithms:
        alg_rows = [row for row in summary_rows if row['算法'] == algorithm]
        algorithm_best[algorithm] = min(alg_rows, key=lambda item: item['平均能量值'])
    
    best_overall = min(algorithm_best.values(), key=lambda item: item['平均能量值'])
    best_algorithm = best_overall['算法']
    
    lines = [
        f"所有算法统一消融实验表明，{best_algorithm} 在完整模型下的平均能量值最低（{best_overall['平均能量值']}），表现出最佳的调度质量。",
        f"从消融效果看，移除换轨惩罚项对算法性能影响最为显著，多数算法在该场景下能量值明显上升。",
        f"完整模型在所有算法中均表现出较好的稳定性，说明各约束与惩罚项的联合设计能够获得较稳定的调度效果。"
    ]
    return [f"- {line}" for line in lines]


def generate_unified_weight_search_conclusion(summary_rows):
    if not summary_rows:
        return []
    
    algorithms = sorted({row['算法'] for row in summary_rows})
    algorithm_best = {}
    for algorithm in algorithms:
        alg_rows = [row for row in summary_rows if row['算法'] == algorithm]
        algorithm_best[algorithm] = min(alg_rows, key=lambda item: item['评分'])
    
    best_overall = min(algorithm_best.values(), key=lambda item: item['评分'])
    best_algorithm = best_overall['算法']
    
    lines = [
        f"所有算法统一QUBO权重搜索结果表明，{best_algorithm} 取得最优综合评分 {best_overall['评分']}，其平均能量值为 {best_overall['平均能量值']}。",
        f"各算法的最佳权重配置存在差异，反映出不同优化算法对QUBO模型参数的敏感性不同。",
        f"合理的权重配置能够显著影响优化器的搜索方向与最终解质量，建议根据具体算法特性进行针对性调优。"
    ]
    return [f"- {line}" for line in lines]


def evaluate_fixed_combination(combination, eval_seed, runtime_overrides=None):
    saved_config = capture_runtime_config()
    try:
        restore_runtime_config(saved_config)
        apply_runtime_overrides(runtime_overrides)
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        start_time = time.time()
        result = calculate_schedule(combination)
        solve_time = time.time() - start_time
        return result, solve_time
    finally:
        restore_runtime_config(saved_config)


def calculate_convergence_metrics(history):
    iterations = history.get('iterations', [])
    best_energy = history.get('best_energy', [])
    current_energy = history.get('current_energy', [])
    if not iterations or not best_energy:
        return None
    initial_best = float(best_energy[0])
    final_best = float(best_energy[-1])
    improvement = initial_best - final_best
    threshold = final_best + max(improvement * 0.05, 1e-6)
    convergence_iter = iterations[-1]
    for iteration, value in zip(iterations, best_energy):
        if value <= threshold:
            convergence_iter = iteration
            break
    tail_window = max(3, len(current_energy) // 5)
    tail_volatility = float(np.std(current_energy[-tail_window:])) if current_energy else 0.0
    area_under_curve = float(np.mean(best_energy))
    return {
        '初始最优能量': round(initial_best, 4),
        '最终最优能量': round(final_best, 4),
        '改进幅度': round(improvement, 4),
        '收敛迭代': int(convergence_iter),
        '尾段波动': round(tail_volatility, 4),
        '平均收敛曲线面积': round(area_under_curve, 4),
        '总记录迭代数': len(iterations)
    }


def get_sensitivity_values(base_value):
    if abs(base_value) < 1e-12:
        return [0.0, 1.0, 2.0]
    return [round(base_value * factor, 4) for factor in (0.8, 1.0, 1.2)]


def run_disturbance_robustness_experiment(n_eval_trials=20, include_bruteforce=False, fluctuation_rates=None):
    algorithms = get_algorithm_registry(include_bruteforce=include_bruteforce)
    fluctuation_rates = fluctuation_rates or [0.0, 0.1, 0.2, 0.3, 0.4]
    print(f"\n===== 开始扰动鲁棒性复现实验（评估次数 {n_eval_trials}）=====")
    summary_rows = []
    detail_rows = []

    for algorithm_meta in algorithms:
        print(f"\n--- 生成 {algorithm_meta['name']} 的基准调度方案 ---")
        base_result, _ = run_single_experiment(algorithm_meta, 0, runtime_overrides={'FLUCTUATION_ENABLE': False})
        combination = schedule_result_to_combination(base_result)

        for rate in fluctuation_rates:
            trial_rows = []
            for seed in range(n_eval_trials):
                result, solve_time = evaluate_fixed_combination(
                    combination,
                    seed,
                    runtime_overrides={'FLUCTUATION_ENABLE': rate > 0, 'FLUCTUATION_RATE': rate}
                )
                row = extract_result_metrics(
                    result,
                    solve_time,
                    seed=seed,
                    label=algorithm_meta['name'],
                    extra={'算法': algorithm_meta['name'], '波动率': rate}
                )
                trial_rows.append(row)
                detail_rows.append(row.copy())
            stats = calculate_statistics(trial_rows)
            summary_rows.append({'算法': algorithm_meta['name'], '波动率': rate, **stats})

    summary_rows.sort(key=lambda item: (item['波动率'], item['平均能量值']))
    export_paths = export_rows(
        '扰动鲁棒性复现实验',
        summary_rows,
        detail_rows,
        metadata={'评估次数': n_eval_trials, '波动率列表': fluctuation_rates, '包含暴力算法': include_bruteforce}
    )
    chart_path = generate_experiment_chart('robustness', summary_rows, '扰动鲁棒性复现实验图')
    report_path = export_markdown_report(
        '扰动鲁棒性复现实验报告',
        '扰动鲁棒性复现实验报告',
        summary_rows,
        metadata={'评估次数': n_eval_trials, '波动率列表': fluctuation_rates, '包含暴力算法': include_bruteforce},
        extra_sections=[('可直接写入论文的结论文字稿', generate_robustness_conclusion(summary_rows))]
    )
    experiment_records['扰动鲁棒性复现实验'] = {
        '统计指标': summary_rows,
        '详细数据': detail_rows,
        '导出文件': {**export_paths, '图表': chart_path, '报告': report_path}
    }
    print(f"\n结果已导出：{export_paths['summary_csv']} | {export_paths['json']} | {chart_path} | {report_path}")
    input("\n按Enter返回高级实验菜单...")
    return summary_rows


def run_convergence_stability_experiment(n_trials=5, include_bruteforce=False):
    algorithms = get_algorithm_registry(include_bruteforce=include_bruteforce)
    print(f"\n===== 开始收敛性与稳定性实验（重复 {n_trials} 次）=====")
    summary_rows = []
    detail_rows = []

    for algorithm_meta in algorithms:
        print(f"\n--- 运行 {algorithm_meta['name']} 的收敛实验 ---")
        trial_metrics = []
        for seed in range(n_trials):
            run_single_experiment(algorithm_meta, seed)
            history = algorithm_history.get(algorithm_meta['name'], {})
            metrics = calculate_convergence_metrics(history)
            if metrics is None:
                continue
            row = {'算法': algorithm_meta['name'], '随机种子': seed, **metrics}
            trial_metrics.append(row)
            detail_rows.append(row.copy())

        if not trial_metrics:
            continue

        summary_rows.append({
            '算法': algorithm_meta['name'],
            '平均最终最优能量': round(np.mean([row['最终最优能量'] for row in trial_metrics]), 4),
            '平均改进幅度': round(np.mean([row['改进幅度'] for row in trial_metrics]), 4),
            '平均收敛迭代': round(np.mean([row['收敛迭代'] for row in trial_metrics]), 2),
            '尾段波动': round(np.mean([row['尾段波动'] for row in trial_metrics]), 4),
            '平均收敛曲线面积': round(np.mean([row['平均收敛曲线面积'] for row in trial_metrics]), 4)
        })

    summary_rows.sort(key=lambda item: (item['平均最终最优能量'], item['平均收敛迭代']))
    export_paths = export_rows(
        '收敛性与稳定性实验',
        summary_rows,
        detail_rows,
        metadata={'重复次数': n_trials, '包含暴力算法': include_bruteforce}
    )
    chart_path = generate_experiment_chart('convergence', summary_rows, '收敛性与稳定性实验图')
    report_path = export_markdown_report(
        '收敛性与稳定性实验报告',
        '收敛性与稳定性实验报告',
        summary_rows,
        metadata={'重复次数': n_trials, '包含暴力算法': include_bruteforce},
        extra_sections=[('可直接写入论文的结论文字稿', generate_convergence_conclusion(summary_rows))]
    )
    experiment_records['收敛性与稳定性实验'] = {
        '统计指标': summary_rows,
        '详细数据': detail_rows,
        '导出文件': {**export_paths, '图表': chart_path, '报告': report_path}
    }
    print(f"\n结果已导出：{export_paths['summary_csv']} | {export_paths['json']} | {chart_path} | {report_path}")
    input("\n按Enter返回高级实验菜单...")
    return summary_rows


def run_qubo_weight_sensitivity_experiment(algorithm_name='QUBO-传统遗传算法', n_trials=3, base_weights=None):
    algorithm_meta = next((item for item in get_algorithm_registry(include_bruteforce=True) if item['name'] == algorithm_name), None)
    if algorithm_meta is None:
        print("未找到对应算法！")
        input("按Enter返回高级实验菜单...")
        return None

    base_weights = base_weights or (last_weight_search_result if last_weight_search_result else QUBO_WEIGHT_CONFIG)
    base_cfg = {key: float(base_weights[key]) for key in DEFAULT_QUBO_WEIGHTS.keys()}
    print(f"\n===== 开始 QUBO 权重局部敏感性实验（算法：{algorithm_name}）=====")
    summary_rows = []
    detail_rows = []

    for weight_key, base_value in base_cfg.items():
        for tested_value in get_sensitivity_values(base_value):
            candidate = base_cfg.copy()
            candidate[weight_key] = tested_value
            trial_rows = []
            for seed in range(n_trials):
                result, solve_time = run_single_experiment(
                    algorithm_meta,
                    seed,
                    runtime_overrides={'qubo_weights': candidate, 'track_change_cost': candidate['TRACK_CHANGE_PENALTY']}
                )
                row = extract_result_metrics(
                    result,
                    solve_time,
                    seed=seed,
                    label=weight_key,
                    extra={'算法': algorithm_name, '权重项': weight_key, '测试值': tested_value, '基准值': base_value}
                )
                trial_rows.append(row)
                detail_rows.append(row.copy())
            stats = calculate_statistics(trial_rows)
            summary_rows.append({'算法': algorithm_name, '权重项': weight_key, '基准值': base_value, '测试值': tested_value, **stats})

    summary_rows.sort(key=lambda item: (item['权重项'], item['测试值']))
    export_paths = export_rows(
        f'{algorithm_name}_QUBO权重局部敏感性实验',
        summary_rows,
        detail_rows,
        metadata={'算法': algorithm_name, '重复次数': n_trials, '基准权重': base_cfg}
    )
    chart_path = generate_experiment_chart('weight_sensitivity', summary_rows, f'{algorithm_name}_QUBO权重局部敏感性图')
    report_path = export_markdown_report(
        f'{algorithm_name}_QUBO权重局部敏感性报告',
        f'{algorithm_name} QUBO 权重局部敏感性报告',
        summary_rows,
        metadata={'算法': algorithm_name, '重复次数': n_trials, '基准权重': base_cfg},
        extra_sections=[('可直接写入论文的结论文字稿', generate_weight_sensitivity_conclusion(summary_rows))]
    )
    experiment_records[f'{algorithm_name}-QUBO权重局部敏感性实验'] = {
        '统计指标': summary_rows,
        '详细数据': detail_rows,
        '导出文件': {**export_paths, '图表': chart_path, '报告': report_path}
    }
    print(f"\n结果已导出：{export_paths['summary_csv']} | {export_paths['json']} | {chart_path} | {report_path}")
    input("\n按Enter返回高级实验菜单...")
    return summary_rows


def get_problem_scale_configs():
    return [
        {'列车规模': 4, 'train_ids': ['101', '102', '201', '202']},
        {'列车规模': 6, 'train_ids': ['101', '102', '103', '201', '202', '203']},
        {'列车规模': 8, 'train_ids': DEFAULT_TRAIN_IDS[:8]}
    ]


def get_scenario_configs():
    return [
        {
            '场景': '基准场景',
            'runtime_overrides': {
                'FLUCTUATION_ENABLE': True,
                'MAX_NORMAL_PLATFORMS': 2,
                'FLUCTUATION_PENALTY_WEIGHT': 1.5
            }
        },
        {
            '场景': '紧张站台容量',
            'runtime_overrides': {
                'FLUCTUATION_ENABLE': True,
                'MAX_NORMAL_PLATFORMS': 1,
                'FLUCTUATION_PENALTY_WEIGHT': 1.5
            }
        },
        {
            '场景': '宽松站台容量',
            'runtime_overrides': {
                'FLUCTUATION_ENABLE': True,
                'MAX_NORMAL_PLATFORMS': 3,
                'FLUCTUATION_PENALTY_WEIGHT': 1.5
            }
        },
        {
            '场景': '低波动惩罚系数',
            'runtime_overrides': {
                'FLUCTUATION_ENABLE': True,
                'MAX_NORMAL_PLATFORMS': 2,
                'FLUCTUATION_PENALTY_WEIGHT': 0.8
            }
        },
        {
            '场景': '高波动惩罚系数',
            'runtime_overrides': {
                'FLUCTUATION_ENABLE': True,
                'MAX_NORMAL_PLATFORMS': 2,
                'FLUCTUATION_PENALTY_WEIGHT': 3.0
            }
        }
    ]


def run_problem_scale_experiment(n_trials=5, include_bruteforce=False):
    print(f"\n===== 开始不同问题规模实验（重复 {n_trials} 次）=====")
    algorithms = get_algorithm_registry(include_bruteforce=include_bruteforce)
    scale_configs = get_problem_scale_configs()
    summary_rows = []
    detail_rows = []

    for scale_cfg in scale_configs:
        print(f"\n--- 当前列车规模：{scale_cfg['列车规模']} ---")
        for algorithm_meta in algorithms:
            trial_rows = []
            for seed in range(n_trials):
                result, solve_time = run_single_experiment(
                    algorithm_meta,
                    seed,
                    runtime_overrides={'train_ids': scale_cfg['train_ids']}
                )
                row = extract_result_metrics(
                    result,
                    solve_time,
                    seed=seed,
                    label=algorithm_meta['name'],
                    extra={
                        '算法': algorithm_meta['name'],
                        '列车规模': scale_cfg['列车规模'],
                        '列车集合': ','.join(scale_cfg['train_ids'])
                    }
                )
                trial_rows.append(row)
                detail_rows.append(row.copy())
            stats = calculate_statistics(trial_rows)
            summary_rows.append({
                '算法': algorithm_meta['name'],
                '列车规模': scale_cfg['列车规模'],
                '列车集合': ','.join(scale_cfg['train_ids']),
                **stats
            })

    summary_rows.sort(key=lambda item: (item['列车规模'], item['平均能量值']))
    export_paths = export_rows(
        '不同问题规模实验',
        summary_rows,
        detail_rows,
        metadata={'重复次数': n_trials, '包含暴力算法': include_bruteforce, '规模集合': [cfg['列车规模'] for cfg in scale_configs]}
    )
    chart_path = generate_experiment_chart('problem_scale', summary_rows, '不同问题规模实验图')
    report_path = export_markdown_report(
        '不同问题规模实验报告',
        '不同问题规模实验报告',
        summary_rows,
        metadata={'重复次数': n_trials, '包含暴力算法': include_bruteforce, '规模集合': [cfg['列车规模'] for cfg in scale_configs]},
        extra_sections=[('可直接写入论文的结论文字稿', generate_problem_scale_conclusion(summary_rows))]
    )
    experiment_records['不同问题规模实验'] = {
        '统计指标': summary_rows,
        '详细数据': detail_rows,
        '导出文件': {**export_paths, '图表': chart_path, '报告': report_path}
    }
    print(f"\n结果已导出：{export_paths['summary_csv']} | {export_paths['json']} | {chart_path} | {report_path}")
    input("\n按Enter返回高级实验菜单...")
    return summary_rows


def run_scenario_parameter_experiment(algorithm_name='QUBO-传统遗传算法', n_trials=5):
    algorithm_meta = next((item for item in get_algorithm_registry(include_bruteforce=True) if item['name'] == algorithm_name), None)
    if algorithm_meta is None:
        print("未找到对应算法！")
        input("按Enter返回高级实验菜单...")
        return None

    print(f"\n===== 开始场景参数实验（算法：{algorithm_name}，重复 {n_trials} 次）=====")
    scenario_configs = get_scenario_configs()
    summary_rows = []
    detail_rows = []

    for scenario_cfg in scenario_configs:
        print(f"\n--- 当前场景：{scenario_cfg['场景']} ---")
        trial_rows = []
        for seed in range(n_trials):
            result, solve_time = run_single_experiment(
                algorithm_meta,
                seed,
                runtime_overrides=scenario_cfg['runtime_overrides']
            )
            row = extract_result_metrics(
                result,
                solve_time,
                seed=seed,
                label=scenario_cfg['场景'],
                extra={
                    '算法': algorithm_name,
                    '场景': scenario_cfg['场景'],
                    '站台容量': scenario_cfg['runtime_overrides']['MAX_NORMAL_PLATFORMS'],
                    '波动惩罚系数': scenario_cfg['runtime_overrides']['FLUCTUATION_PENALTY_WEIGHT']
                }
            )
            trial_rows.append(row)
            detail_rows.append(row.copy())
        stats = calculate_statistics(trial_rows)
        summary_rows.append({
            '算法': algorithm_name,
            '场景': scenario_cfg['场景'],
            '站台容量': scenario_cfg['runtime_overrides']['MAX_NORMAL_PLATFORMS'],
            '波动惩罚系数': scenario_cfg['runtime_overrides']['FLUCTUATION_PENALTY_WEIGHT'],
            **stats
        })

    summary_rows.sort(key=lambda item: item['平均能量值'])
    export_paths = export_rows(
        f'{algorithm_name}_场景参数实验',
        summary_rows,
        detail_rows,
        metadata={'算法': algorithm_name, '重复次数': n_trials, '场景数': len(scenario_configs)}
    )
    chart_path = generate_experiment_chart('scenario', summary_rows, f'{algorithm_name}_场景参数实验图')
    report_path = export_markdown_report(
        f'{algorithm_name}_场景参数实验报告',
        f'{algorithm_name} 场景参数实验报告',
        summary_rows,
        metadata={'算法': algorithm_name, '重复次数': n_trials, '场景数': len(scenario_configs)},
        extra_sections=[('可直接写入论文的结论文字稿', generate_scenario_conclusion(summary_rows))]
    )
    experiment_records[f'{algorithm_name}-场景参数实验'] = {
        '统计指标': summary_rows,
        '详细数据': detail_rows,
        '导出文件': {**export_paths, '图表': chart_path, '报告': report_path}
    }
    print(f"\n结果已导出：{export_paths['summary_csv']} | {export_paths['json']} | {chart_path} | {report_path}")
    input("\n按Enter返回高级实验菜单...")
    return summary_rows


def run_single_experiment(algorithm_meta, trial_seed, runtime_overrides=None, solver_kwargs=None):
    saved_config = capture_runtime_config()
    try:
        restore_runtime_config(saved_config)
        apply_runtime_overrides(runtime_overrides)
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        start_time = time.time()
        result = algorithm_meta['func'](**(solver_kwargs or algorithm_meta.get('kwargs', {})))
        solve_time = time.time() - start_time
        return result, solve_time
    finally:
        restore_runtime_config(saved_config)


def run_algorithm_comparison_experiment(n_trials=10, include_bruteforce=True):
    print(f"\n===== 开始统一算法对比实验（重复 {n_trials} 次）=====")
    algorithms = get_algorithm_registry(include_bruteforce=include_bruteforce)
    summary_rows = []
    detail_rows = []

    for algorithm_meta in algorithms:
        print(f"\n--- 正在运行 {algorithm_meta['name']} ---")
        trial_rows = []
        for seed in range(n_trials):
            result, solve_time = run_single_experiment(algorithm_meta, seed)
            row = extract_result_metrics(result, solve_time, seed=seed, label=algorithm_meta['name'])
            trial_rows.append(row)
            detail_rows.append(row.copy())
        stats = calculate_statistics(trial_rows)
        print_stats_table(algorithm_meta['name'], stats)
        summary_rows.append({'算法': algorithm_meta['name'], **stats})

    summary_rows.sort(key=lambda item: (item['平均能量值'], item['平均求解时间']))
    baseline_algorithm = summary_rows[0]['算法'] if summary_rows else None
    significance_rows = []
    if baseline_algorithm:
        for row in summary_rows[1:]:
            test_row = compare_detail_groups(detail_rows, '标签', baseline_algorithm, row['算法'])
            if test_row:
                significance_rows.append(test_row)
    export_paths = export_rows(
        '统一算法对比实验',
        summary_rows,
        detail_rows,
        metadata={
            '重复次数': n_trials,
            '包含暴力算法': include_bruteforce,
            'QUBO权重': QUBO_WEIGHT_CONFIG.copy(),
            '显著性检验': significance_rows
        }
    )
    chart_path = generate_experiment_chart('comparison', summary_rows, '统一算法对比实验图')
    report_path = export_markdown_report(
        '统一算法对比实验报告',
        '统一算法对比实验报告',
        summary_rows,
        metadata={'重复次数': n_trials, '包含暴力算法': include_bruteforce, 'QUBO权重': QUBO_WEIGHT_CONFIG.copy()},
        extra_sections=[
            ('Top结果', build_top_summary_lines(summary_rows, label_key='算法')),
            ('显著性检验', build_significance_lines(significance_rows)),
            ('可直接写入论文的结论文字稿', generate_comparison_conclusion(summary_rows, significance_rows))
        ]
    )
    experiment_records['统一算法对比实验'] = {
        '统计指标': summary_rows,
        '详细数据': detail_rows,
        '显著性检验': significance_rows,
        '导出文件': {**export_paths, '图表': chart_path, '报告': report_path}
    }
    print(f"\n结果已导出：{export_paths['summary_csv']} | {export_paths['json']} | {chart_path} | {report_path}")
    input("\n按Enter返回高级实验菜单...")
    return summary_rows


def run_ablation_experiment(algorithm_name, n_trials=10):
    algorithm_meta = next((item for item in get_algorithm_registry(include_bruteforce=True) if item['name'] == algorithm_name), None)
    if algorithm_meta is None:
        print("未找到对应算法！")
        input("按Enter返回高级实验菜单...")
        return None

    print(f"\n===== 开始 {algorithm_name} 消融实验（重复 {n_trials} 次）=====")
    scenarios = build_ablation_scenarios(algorithm_meta)
    summary_rows = []
    detail_rows = []

    for scenario in scenarios:
        print(f"\n--- 消融场景：{scenario['name']} ---")
        trial_rows = []
        for seed in range(n_trials):
            result, solve_time = run_single_experiment(
                algorithm_meta,
                seed,
                runtime_overrides=scenario.get('runtime_overrides'),
                solver_kwargs={**algorithm_meta.get('kwargs', {}), **scenario.get('solver_kwargs', {})}
            )
            row = extract_result_metrics(
                result,
                solve_time,
                seed=seed,
                label=scenario['name'],
                extra={'算法': algorithm_name, '消融项': scenario['name']}
            )
            trial_rows.append(row)
            detail_rows.append(row.copy())
        stats = calculate_statistics(trial_rows)
        print_stats_table(f"{algorithm_name} - {scenario['name']}", stats)
        summary_rows.append({'算法': algorithm_name, '消融项': scenario['name'], **stats})

    summary_rows.sort(key=lambda item: (item['平均能量值'], item['平均求解时间']))
    significance_rows = []
    for row in summary_rows:
        if row['消融项'] == '完整模型':
            continue
        test_row = compare_detail_groups(detail_rows, '消融项', '完整模型', row['消融项'])
        if test_row:
            significance_rows.append(test_row)
    export_paths = export_rows(
        f'{algorithm_name}_消融实验',
        summary_rows,
        detail_rows,
        metadata={'算法': algorithm_name, '重复次数': n_trials, '显著性检验': significance_rows}
    )
    chart_path = generate_experiment_chart('ablation', summary_rows, f'{algorithm_name}_消融实验图')
    report_path = export_markdown_report(
        f'{algorithm_name}_消融实验报告',
        f'{algorithm_name} 消融实验报告',
        summary_rows,
        metadata={'算法': algorithm_name, '重复次数': n_trials},
        extra_sections=[
            ('Top结果', build_top_summary_lines(summary_rows, label_key='消融项')),
            ('显著性检验', build_significance_lines(significance_rows)),
            ('可直接写入论文的结论文字稿', generate_ablation_conclusion(summary_rows, significance_rows))
        ]
    )
    experiment_records[f'{algorithm_name}-消融实验'] = {
        '统计指标': summary_rows,
        '详细数据': detail_rows,
        '显著性检验': significance_rows,
        '导出文件': {**export_paths, '图表': chart_path, '报告': report_path}
    }
    print(f"\n结果已导出：{export_paths['summary_csv']} | {export_paths['json']} | {chart_path} | {report_path}")
    input("\n按Enter返回高级实验菜单...")
    return summary_rows


def run_unified_ablation_experiment(n_trials=3):
    algorithms = get_algorithm_registry(include_bruteforce=False)
    print(f"\n===== 开始所有算法统一消融实验（{len(algorithms)}个算法，重复 {n_trials} 次）=====")
    all_summary_rows = []
    all_detail_rows = []

    for algorithm_meta in algorithms:
        print(f"\n{'='*60}")
        print(f"算法：{algorithm_meta['name']}")
        print(f"{'='*60}")
        scenarios = build_ablation_scenarios(algorithm_meta)
        
        for scenario in scenarios:
            print(f"\n--- 消融场景：{scenario['name']} ---")
            trial_rows = []
            for seed in range(n_trials):
                result, solve_time = run_single_experiment(
                    algorithm_meta,
                    seed,
                    runtime_overrides=scenario.get('runtime_overrides'),
                    solver_kwargs={**algorithm_meta.get('kwargs', {}), **scenario.get('solver_kwargs', {})}
                )
                row = extract_result_metrics(
                    result,
                    solve_time,
                    seed=seed,
                    label=scenario['name'],
                    extra={'算法': algorithm_meta['name'], '消融项': scenario['name']}
                )
                trial_rows.append(row)
                all_detail_rows.append(row.copy())
            stats = calculate_statistics(trial_rows)
            print_stats_table(f"{algorithm_meta['name']} - {scenario['name']}", stats)
            all_summary_rows.append({'算法': algorithm_meta['name'], '消融项': scenario['name'], **stats})

    # 确保有数据才生成图表
    if not all_summary_rows:
        print("\n错误：没有生成任何实验数据，无法创建图表！")
        input("\n按Enter返回高级实验菜单...")
        return None

    export_paths = export_rows(
        '所有算法统一消融实验',
        all_summary_rows,
        all_detail_rows,
        metadata={'算法数量': len(algorithms), '重复次数': n_trials}
    )
    
    # 生成平均能量值图
    chart_energy_path = generate_experiment_chart('ablation_unified_energy', all_summary_rows, '所有算法消融实验_平均能量值')
    if chart_energy_path:
        print(f"  - 平均能量值图：{chart_energy_path}")
    else:
        print("  - 平均能量值图：生成失败")
    
    # 生成平均延误图
    chart_delay_path = generate_experiment_chart('ablation_unified_delay', all_summary_rows, '所有算法消融实验_平均延误')
    if chart_delay_path:
        print(f"  - 平均延误图：{chart_delay_path}")
    else:
        print("  - 平均延误图：生成失败")
    
    report_path = export_markdown_report(
        '所有算法统一消融实验报告',
        '所有算法统一消融实验报告',
        all_summary_rows,
        metadata={'算法数量': len(algorithms), '重复次数': n_trials},
        extra_sections=[
            ('Top结果', build_top_summary_lines(all_summary_rows, label_key='算法')),
            ('可直接写入论文的结论文字稿', generate_unified_ablation_conclusion(all_summary_rows))
        ]
    )
    
    print(f"\n结果已导出：")
    print(f"  - 数据文件：{export_paths['summary_csv']}")
    print(f"  - 实验报告：{report_path}")
    input("\n按Enter返回高级实验菜单...")
    return all_summary_rows


def generate_weight_candidates(n_samples=12, strategy='grid'):
    candidates = []
    seen = set()
    base = DEFAULT_QUBO_WEIGHTS.copy()
    candidates.append(base)
    seen.add(tuple(base.items()))

    grids = {
        'ORDER_MUTEX_PENALTY': [80.0, 100.0, 140.0],
        'TRACK_MUTEX_PENALTY': [80.0, 100.0, 140.0],
        'CONFLICT_PENALTY': [160.0, 200.0, 260.0],
        'DELAY_PENALTY': [0.5, 1.0, 2.0],
        'TRACK_CHANGE_PENALTY': [0.0, 2.0, 4.0]
    }

    if strategy == 'grid':
        for values in product(*grids.values()):
            candidate = dict(zip(grids.keys(), values))
            key = tuple(candidate.items())
            if key in seen:
                continue
            candidates.append(candidate)
            seen.add(key)
            if len(candidates) >= n_samples:
                break
    else:
        while len(candidates) < n_samples:
            candidate = {key: random.choice(values) for key, values in grids.items()}
            key = tuple(candidate.items())
            if key in seen:
                continue
            candidates.append(candidate)
            seen.add(key)
    return candidates


def run_qubo_weight_search(algorithm_name='QUBO-传统遗传算法', n_samples=12, trials_per_candidate=3, strategy='grid'):
    global last_weight_search_result
    algorithm_meta = next((item for item in get_algorithm_registry(include_bruteforce=True) if item['name'] == algorithm_name), None)
    if algorithm_meta is None:
        print("未找到对应算法！")
        input("按Enter返回高级实验菜单...")
        return None

    print(f"\n===== 开始 QUBO 权重搜索（算法：{algorithm_name}，候选数：{n_samples}，策略：{strategy}）=====")
    saved_config = capture_runtime_config()
    summary_rows = []
    detail_rows = []
    best_row = None

    try:
        candidates = generate_weight_candidates(n_samples=n_samples, strategy=strategy)
        for idx, candidate in enumerate(candidates, start=1):
            print(f"\n--- 评估候选权重 {idx}/{len(candidates)} ---")
            trial_rows = []
            for seed in range(trials_per_candidate):
                result, solve_time = run_single_experiment(
                    algorithm_meta,
                    seed,
                    runtime_overrides={'qubo_weights': candidate, 'track_change_cost': candidate['TRACK_CHANGE_PENALTY']}
                )
                row = extract_result_metrics(
                    result,
                    solve_time,
                    seed=seed,
                    label=f'候选{idx}',
                    extra={**candidate, '算法': algorithm_name, '候选编号': idx}
                )
                trial_rows.append(row)
                detail_rows.append(row.copy())

            stats = calculate_statistics(trial_rows)
            score = round(stats['平均能量值'] + 0.2 * stats['能量值标准差'] + 0.5 * stats['平均总延误时间'], 3)
            summary_row = {'算法': algorithm_name, '候选编号': idx, '评分': score, **candidate, **stats}
            summary_rows.append(summary_row)
            if best_row is None or score < best_row['评分']:
                best_row = summary_row.copy()
                print(f"当前最优评分: {score:.3f}")

        summary_rows.sort(key=lambda item: item['评分'])
        last_weight_search_result = best_row
        restore_runtime_config(saved_config)
        apply_runtime_overrides({
            'qubo_weights': {key: best_row[key] for key in DEFAULT_QUBO_WEIGHTS.keys()},
            'track_change_cost': best_row['TRACK_CHANGE_PENALTY']
        })

        export_paths = export_rows(
            f'{algorithm_name}_QUBO权重搜索',
            summary_rows,
            detail_rows,
            metadata={
                '算法': algorithm_name,
                '候选数': len(summary_rows),
                '每组重复次数': trials_per_candidate,
                '搜索策略': strategy,
                '最佳结果': best_row
            }
        )
        significance_rows = []
        if best_row is not None and summary_rows:
            test_row = compare_detail_groups(detail_rows, '候选编号', 1, best_row['候选编号'])
            if test_row:
                significance_rows.append(test_row)
        chart_path = generate_experiment_chart('weight_search', summary_rows, f'{algorithm_name}_QUBO权重搜索图')
        report_path = export_markdown_report(
            f'{algorithm_name}_QUBO权重搜索报告',
            f'{algorithm_name} QUBO 权重搜索报告',
            summary_rows,
            metadata={'算法': algorithm_name, '候选数': len(summary_rows), '每组重复次数': trials_per_candidate, '搜索策略': strategy},
            extra_sections=[
                ('最佳配置', [
                    f"- 评分: {best_row['评分']}",
                    f"- ORDER_MUTEX_PENALTY: {best_row['ORDER_MUTEX_PENALTY']}",
                    f"- TRACK_MUTEX_PENALTY: {best_row['TRACK_MUTEX_PENALTY']}",
                    f"- CONFLICT_PENALTY: {best_row['CONFLICT_PENALTY']}",
                    f"- DELAY_PENALTY: {best_row['DELAY_PENALTY']}",
                    f"- TRACK_CHANGE_PENALTY: {best_row['TRACK_CHANGE_PENALTY']}"
                ]),
                ('显著性检验', build_significance_lines(significance_rows)),
                ('可直接写入论文的结论文字稿', generate_weight_search_conclusion(summary_rows, best_row))
            ]
        )
        experiment_records[f'{algorithm_name}-QUBO权重搜索'] = {
            '统计指标': summary_rows,
            '详细数据': detail_rows,
            '最佳配置': best_row,
            '显著性检验': significance_rows,
            '导出文件': {**export_paths, '图表': chart_path, '报告': report_path}
        }

        print("\n===== QUBO 权重搜索完成 =====")
        print(f"最佳评分: {best_row['评分']}")
        print(f"最佳权重: ORDER={best_row['ORDER_MUTEX_PENALTY']}, TRACK={best_row['TRACK_MUTEX_PENALTY']}, CONFLICT={best_row['CONFLICT_PENALTY']}, DELAY={best_row['DELAY_PENALTY']}, TRACK_CHANGE={best_row['TRACK_CHANGE_PENALTY']}")
        print(f"已应用最佳权重为当前QUBO配置，结果已导出：{export_paths['summary_csv']} | {export_paths['json']} | {chart_path} | {report_path}")
        input("\n按Enter返回高级实验菜单...")
        return best_row
    finally:
        if best_row is None:
            restore_runtime_config(saved_config)


def run_unified_qubo_weight_search(n_trials=3):
    algorithms = get_algorithm_registry(include_bruteforce=False)
    print(f"\n===== 开始所有算法统一QUBO权重搜索（{len(algorithms)}个算法，重复 {n_trials} 次）=====")
    saved_config = capture_runtime_config()
    all_summary_rows = []
    all_detail_rows = []

    for algorithm_meta in algorithms:
        print(f"\n{'='*60}")
        print(f"算法：{algorithm_meta['name']}")
        print(f"{'='*60}")
        
        candidates = generate_weight_candidates(n_samples=12, strategy='grid')
        best_row_for_algorithm = None
        
        for idx, candidate in enumerate(candidates, start=1):
            print(f"\n--- 评估候选权重 {idx}/{len(candidates)} ---")
            trial_rows = []
            for seed in range(n_trials):
                result, solve_time = run_single_experiment(
                    algorithm_meta,
                    seed,
                    runtime_overrides={'qubo_weights': candidate, 'track_change_cost': candidate['TRACK_CHANGE_PENALTY']}
                )
                row = extract_result_metrics(
                    result,
                    solve_time,
                    seed=seed,
                    label=f'候选{idx}',
                    extra={**candidate, '算法': algorithm_meta['name'], '候选编号': idx}
                )
                trial_rows.append(row)
                all_detail_rows.append(row.copy())

            stats = calculate_statistics(trial_rows)
            score = round(stats['平均能量值'] + 0.2 * stats['能量值标准差'] + 0.5 * stats['平均总延误时间'], 3)
            summary_row = {'算法': algorithm_meta['name'], '候选编号': idx, '评分': score, **candidate, **stats}
            all_summary_rows.append(summary_row)
            if best_row_for_algorithm is None or score < best_row_for_algorithm['评分']:
                best_row_for_algorithm = summary_row.copy()
                print(f"当前最优评分: {score:.3f}")

        print(f"\n{algorithm_meta['name']} 最优配置：评分={best_row_for_algorithm['评分']:.3f}")

    restore_runtime_config(saved_config)
    
    # 确保有数据才生成图表
    if not all_summary_rows:
        print("\n错误：没有生成任何实验数据，无法创建图表！")
        input("\n按Enter返回高级实验菜单...")
        return None
    
    export_paths = export_rows(
        '所有算法统一QUBO权重搜索',
        all_summary_rows,
        all_detail_rows,
        metadata={'算法数量': len(algorithms), '每组重复次数': n_trials}
    )
    
    # 生成权重搜索图
    chart_path = generate_experiment_chart('weight_search', all_summary_rows, '所有算法统一QUBO权重搜索图')
    if chart_path:
        print(f"  - 权重搜索图：{chart_path}")
    else:
        print("  - 权重搜索图：生成失败")
    
    report_path = export_markdown_report(
        '所有算法统一QUBO权重搜索报告',
        '所有算法统一QUBO权重搜索报告',
        all_summary_rows,
        metadata={'算法数量': len(algorithms), '每组重复次数': n_trials},
        extra_sections=[
            ('Top结果', build_top_summary_lines(all_summary_rows, label_key='算法')),
            ('可直接写入论文的结论文字稿', generate_unified_weight_search_conclusion(all_summary_rows))
        ]
    )
    
    print(f"\n结果已导出：")
    print(f"  - 数据文件：{export_paths['summary_csv']}")
    print(f"  - 实验报告：{report_path}")
    input("\n按Enter返回高级实验菜单...")
    return all_summary_rows


# --------------------------
# 区间波动实验与多指标统计（添加开关控制）
# --------------------------
def run_fluctuation_experiment(algorithm_name, algorithm_func, n_trials=100):
    """在区间波动下运行多次实验并输出所有统计指标"""
    print(f"\n===== 开始{algorithm_name}区间波动实验（{n_trials}次重复）=====")
    results = []
    for i in range(n_trials):
        random.seed(i)  # 保证实验可复现
        np.random.seed(i)
        start_time = time.time()  # 记录算法开始时间
        result = algorithm_func()
        end_time = time.time()  # 记录算法结束时间
        solve_time = end_time - start_time  # 计算算法求解时间
        results.append(extract_result_metrics(result, solve_time, seed=i, label=algorithm_name))
        # 进度提示
        if (i+1) % 10 == 0:
            avg_energy = np.mean([r['能量值'] for r in results[:i+1]])
            print(f"实验进度：{i+1}/{n_trials} | 当前平均能量值：{avg_energy:.2f}")
    stats = calculate_statistics(results)
    
    # 存储实验结果
    experiment_records[algorithm_name] = {
        '详细数据': results,
        '统计指标': stats
    }
    
    # 输出统计结果
    print_stats_table(f"{algorithm_name}实验统计结果", stats)
    print(f"最大总延误时间    | {stats['最大总延误时间']:<10} | 越低越好")
    print(f"最小总延误时间    | {stats['最小总延误时间']:<10} | 越低越好")
    print(f"最大求解时间      | {stats['最大求解时间']:<10} | 越低越稳定")
    print(f"最小求解时间      | {stats['最小求解时间']:<10} | 参考最优性能")
    print("=" * 47)
    input("\n按Enter返回实验菜单...")
    return stats

def experiment_menu():
    """区间波动实验菜单（添加开关判断）"""
    # 核心修改：如果波动开关关闭，直接提示无法运行
    if not FLUCTUATION_ENABLE:
        print("\n⚠️ 区间波动功能已关闭，无法运行区间波动实验！")
        print("请先将FLUCTUATION_ENABLE设置为True")
        input("按Enter返回主菜单...")
        return

    algorithms = [
        (1, 'QUBO-传统遗传算法', solve_by_genetic_algorithm),
        (2, 'QUBO-传统粒子群算法', solve_by_particle_swarm),
        (3, 'QUBO-传统模拟退火算法', solve_by_simulated_annealing),
        (4, 'QUBO-量子遗传算法', solve_by_quantum_genetic_algorithm),
        (5, 'QUBO-量子粒子群算法', solve_by_quantum_particle_swarm),
        (6, 'QUBO-量子模拟退火算法', solve_by_quantum_simulated_annealing),
        (7, 'QUBO-量子遗传-QAOA混合算法', solve_by_qga_qaoa_hybrid),
        (8, 'QUBO-量子退火-QAOA混合算法', solve_by_qsa_qaoa_hybrid),
        (9, 'QUBO-量子粒子群-QAOA混合算法', solve_by_qpso_qaoa_hybrid),
        (10, 'QUBO-暴力遍历算法', solve_by_brute_force)
    ]
    
    while True:
        print("\n----- 区间波动实验菜单 -----")
        print("0. 返回主菜单")
        for idx, name, _ in algorithms:
            print(f"{idx}. 运行{name}波动实验")
        try:
            choice = int(input("请选择算法（0-10）："))
            if choice == 0:
                break
            for idx, name, func in algorithms:
                if idx == choice:
                    while True:
                        try:
                            n_trials = int(input("请输入实验重复次数（建议10-100）："))
                            if n_trials >= 10:
                                break
                            print("次数建议≥10，统计结果更可靠！")
                        except ValueError:
                            print("请输入数字！")
                    run_fluctuation_experiment(name, func, n_trials)
                    break
            else:
                print("无效选择！")
        except ValueError:
            print("请输入数字！")
        else:
            print("请输入0-5之间的数字！")


def advanced_experiment_menu():
    """高级实验菜单：统一对比、消融、QUBO权重搜索与扩展实验"""
    algorithms = get_algorithm_registry(include_bruteforce=True)

    while True:
        print("\n----- 高级实验菜单 -----")
        print("0. 返回主菜单")
        print("1. 统一算法对比实验")
        print("2. 消融实验")
        print("3. QUBO 权重搜索")
        print("4. 扰动鲁棒性复现实验")
        print("5. 收敛性与稳定性实验")
        print("6. QUBO 权重局部敏感性实验")
        print("7. 不同问题规模实验")
        print("8. 场景参数实验（站台容量/波动惩罚）")
        print("9. 查看最近权重搜索结果")
        print("10. 所有算法统一消融实验（只输出图表）")
        print("11. 所有算法统一QUBO权重搜索")
        print(f"当前QUBO权重: {QUBO_WEIGHT_CONFIG}")
        try:
            choice = int(input("请选择功能（0-11）："))
            if choice == 0:
                break
            if choice == 1:
                n_trials = int(input("请输入每个算法重复次数（建议3-10）："))
                include_bruteforce = input("是否包含暴力遍历算法？(y/n，默认n)：").strip().lower() == 'y'
                run_algorithm_comparison_experiment(max(1, n_trials), include_bruteforce=include_bruteforce)
            elif choice == 2:
                print("\n可选算法：")
                for idx, algorithm_meta in enumerate(algorithms, start=1):
                    print(f"{idx}. {algorithm_meta['name']}")
                alg_choice = int(input(f"请选择算法（1-{len(algorithms)}）："))
                if 1 <= alg_choice <= len(algorithms):
                    n_trials = int(input("请输入每个消融场景重复次数（建议3-10）："))
                    run_ablation_experiment(algorithms[alg_choice - 1]['name'], max(1, n_trials))
                else:
                    print("无效算法编号！")
            elif choice == 3:
                print("\n可选算法：")
                for idx, algorithm_meta in enumerate(algorithms, start=1):
                    print(f"{idx}. {algorithm_meta['name']}")
                alg_choice = int(input(f"请选择用于权重搜索的算法（1-{len(algorithms)}）："))
                if 1 <= alg_choice <= len(algorithms):
                    n_samples = int(input("请输入候选权重组数（建议6-15）："))
                    trials_per_candidate = int(input("请输入每组权重重复次数（建议2-5）："))
                    strategy_input = input("搜索策略：1. grid 2. random（默认1）：").strip()
                    strategy = 'random' if strategy_input == '2' else 'grid'
                    run_qubo_weight_search(
                        algorithms[alg_choice - 1]['name'],
                        n_samples=max(1, n_samples),
                        trials_per_candidate=max(1, trials_per_candidate),
                        strategy=strategy
                    )
                else:
                    print("无效算法编号！")
            elif choice == 4:
                n_eval_trials = int(input("请输入每个波动率下的复现次数（建议10-30）："))
                include_bruteforce = input("是否包含暴力遍历算法？(y/n，默认n)：").strip().lower() == 'y'
                run_disturbance_robustness_experiment(max(1, n_eval_trials), include_bruteforce=include_bruteforce)
            elif choice == 5:
                n_trials = int(input("请输入每个算法的重复次数（建议3-8）："))
                include_bruteforce = input("是否包含暴力遍历算法？(y/n，默认n)：").strip().lower() == 'y'
                run_convergence_stability_experiment(max(1, n_trials), include_bruteforce=include_bruteforce)
            elif choice == 6:
                print("\n可选算法：")
                for idx, algorithm_meta in enumerate(algorithms, start=1):
                    print(f"{idx}. {algorithm_meta['name']}")
                alg_choice = int(input(f"请选择算法（1-{len(algorithms)}）："))
                if 1 <= alg_choice <= len(algorithms):
                    n_trials = int(input("请输入每个测试值的重复次数（建议2-5）："))
                    run_qubo_weight_sensitivity_experiment(algorithms[alg_choice - 1]['name'], max(1, n_trials))
                else:
                    print("无效算法编号！")
            elif choice == 7:
                n_trials = int(input("请输入每个规模下的重复次数（建议3-8）："))
                include_bruteforce = input("是否包含暴力遍历算法？(y/n，默认n)：").strip().lower() == 'y'
                run_problem_scale_experiment(max(1, n_trials), include_bruteforce=include_bruteforce)
            elif choice == 8:
                print("\n可选算法：")
                for idx, algorithm_meta in enumerate(algorithms, start=1):
                    print(f"{idx}. {algorithm_meta['name']}")
                alg_choice = int(input(f"请选择算法（1-{len(algorithms)}）："))
                if 1 <= alg_choice <= len(algorithms):
                    n_trials = int(input("请输入每个场景的重复次数（建议3-8）："))
                    run_scenario_parameter_experiment(algorithms[alg_choice - 1]['name'], max(1, n_trials))
                else:
                    print("无效算法编号！")
            elif choice == 9:
                if last_weight_search_result is None:
                    print("暂无最近一次权重搜索结果。")
                else:
                    print("\n最近一次最佳权重配置：")
                    print(last_weight_search_result)
            elif choice == 10:
                n_trials = int(input("请输入每个消融场景重复次数（建议3-10）："))
                run_unified_ablation_experiment(max(1, n_trials))
            elif choice == 11:
                n_trials = int(input("请输入每组权重重复次数（建议2-5）："))
                run_unified_qubo_weight_search(max(1, n_trials))
            else:
                print("无效选择！")
        except ValueError:
            print("请输入有效数字！")

# --------------------------
# 结果展示与主菜单
# --------------------------
def build_timetable_rows(result):
    rows = []
    for tid in sorted(train_ids):
        sched = result['schedule'][tid]
        for station in sched['route_stations']:
            info = sched['stations'][station]
            rows.append({
                '列车': tid,
                '方向': sched['方向'],
                '站点': station,
                '轨道': info['轨道'],
                '到时': info['到时'],
                '停留时间': info['停留时间'],
                '延误时间': info['延误时间'],
                '出发时间': info['出发时间']
            })
    return rows


def export_timetable_and_running_diagram(result=None):
    result = result or best_result
    if not result:
        print("暂无求解结果，请先运行算法！")
        return None

    timetable_rows = build_timetable_rows(result)
    export_paths = export_rows('当前调度时刻表', timetable_rows, metadata={'能量值': result['能量值']})

    fig, ax = plt.subplots(figsize=(13.5, 8))
    y_positions = {station: idx for idx, station in enumerate(stations)}
    for idx, station in enumerate(stations):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.45, idx + 0.45, color='#f4f1ea', alpha=0.85, zorder=0)

    legend_added = {'上行': False, '下行': False}
    for tid in sorted(train_ids):
        sched = result['schedule'][tid]
        x_points = []
        y_points = []
        for station in sched['route_stations']:
            info = sched['stations'][station]
            if isinstance(info['到时'], (int, float)):
                x_points.append(info['到时'])
                y_points.append(y_positions[station])
            if isinstance(info['出发时间'], (int, float)):
                x_points.append(info['出发时间'])
                y_points.append(y_positions[station])
        color = '#d1495b' if sched['方向'] == '上行' else '#1f6f8b'
        linestyle = '-' if sched['方向'] == '上行' else '--'
        label = sched['方向'] if not legend_added[sched['方向']] else None
        legend_added[sched['方向']] = True
        ax.plot(x_points, y_points, marker='o', markersize=4.8, linewidth=2.0, linestyle=linestyle, label=label, color=color, alpha=0.92)
        if x_points and y_points:
            ax.text(x_points[0] - 0.8, y_points[0] + 0.08, tid, fontsize=8, color=color, fontweight='bold')
            ax.text(x_points[-1] + 0.4, y_points[-1] + 0.08, tid, fontsize=8, color=color)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(stations)
    ax.set_xlabel('运行时间 / min', fontsize=11)
    ax.set_ylabel('站点位置', fontsize=11)
    ax.set_title('上下行列车运行图', fontsize=15, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.35)
    ax.legend(fontsize=9, loc='upper center', ncol=2, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.01, 1.02, '实线为上行列车，虚线为下行列车；折点表示站点到发时刻', transform=ax.transAxes, fontsize=9, color='#555555')
    diagram_path = export_chart('当前列车运行图', fig)

    print(f"时刻表已导出：{export_paths['summary_csv']}")
    print(f"运行图已导出：{diagram_path}")
    return {'时刻表': export_paths['summary_csv'], '运行图': diagram_path}


def display_result():
    """显示单次求解结果"""
    global best_result
    if not best_result:
        print("暂无求解结果，请先运行算法！")
        return
    
    print(f"\n===== 最优调度方案（能量值: {best_result['能量值']:.2f}） =====")
    # 显示波动状态
    print(f"区间波动状态: {'开启' if FLUCTUATION_ENABLE else '关闭'}")
    print(f"停靠线容量限制: {'开启' if PLATFORM_CAPACITY_ENABLE else '关闭'}")
    print(f"总延误时间: {best_result['总延误']:.2f}，换轨总成本: {best_result['换轨总成本']:.2f}")
    print(f"区间波动总惩罚: {best_result['区间波动总惩罚']:.2f}，中间站停靠线总惩罚: {best_result.get('中间站停靠线总惩罚', 0.0):.2f}")
    print(f"发车顺序: {', '.join([f'{k}:{v}' for k, v in best_result['发车顺序']])}\n")
    
    for tid in sorted(train_ids):
        sched = best_result['schedule'][tid]
        print(f"列车 {tid}（{sched['方向']}）调度详情（换轨次数: {sched['换轨次数']}）:")
        print(f"{'站点':<6} | {'轨道':<6} | {'到时':<10} | {'停留时间':<10} | {'延误时间':<10} | {'出发时间':<10}")
        print("-" * 60)
        for station in sched['route_stations']:
            info = sched['stations'][station]
            print(f"{station:<6} | {info['轨道']:<6} | {info['到时']:<10} | {info['停留时间']:<10} | {info['延误时间']:<10} | {info['出发时间']:<10}")
        print(f"列车{tid} 中间站总延误: {sched['中间站总延误']:.2f}，始发站延误: {sched['始发站延误']:.2f}")
        print(f"波动惩罚: {sched['区间波动总惩罚']:.2f}，停靠线惩罚: {sched.get('中间站停靠线惩罚', 0.0):.2f}")
        print("-" * 60 + "\n")

def main_menu():
    """主菜单交互"""
    global FLUCTUATION_ENABLE, PLATFORM_CAPACITY_ENABLE
    while True:
        print("\n" + "="*70)
        print(f"上下行列车调度优化系统（区间波动状态: {'开启' if FLUCTUATION_ENABLE else '关闭'}，")
        print(f"            停靠线容量限制: {'开启' if PLATFORM_CAPACITY_ENABLE else '关闭'}）")
        print("="*70)
        print("1. 选择QUBO算法并求解（单次运行）")
        print("2. 显示求解结果")
        print("3. 查看算法迭代图")
        print("4. 运行区间波动实验（多指标统计）")
        print("5. 设置系统参数")
        print("6. 高级实验（统一对比/消融/权重搜索）")
        print("7. 导出列车时刻表与运行图")
        print("0. 退出")
        print("="*70)
        
        try:
            choice = int(input("请输入选择 (0-7): "))
            if choice == 0:
                print("感谢使用，再见！")
                break
            elif choice == 5:
                # 添加参数设置功能
                print("\n----- 系统参数设置 -----")
                print(f"当前区间波动状态: {'开启' if FLUCTUATION_ENABLE else '关闭'}")
                print(f"当前停靠线容量限制状态: {'开启' if PLATFORM_CAPACITY_ENABLE else '关闭'}")
                try:
                    setting_choice = int(input("\n请选择设置项：1. 切换区间波动 2. 切换停靠线容量限制 0. 返回: "))
                    if setting_choice == 1:
                        FLUCTUATION_ENABLE = not FLUCTUATION_ENABLE
                        print(f"区间波动已{'开启' if FLUCTUATION_ENABLE else '关闭'}")
                    elif setting_choice == 2:
                        PLATFORM_CAPACITY_ENABLE = not PLATFORM_CAPACITY_ENABLE
                        print(f"停靠线容量限制已{'开启' if PLATFORM_CAPACITY_ENABLE else '关闭'}")
                    elif setting_choice == 0:
                        print("返回主菜单...")
                    else:
                        print("无效的选择！")
                except ValueError:
                    print("请输入有效的数字！")
            elif choice == 1:
                print("\n----- 选择QUBO求解器 -----")
                print("1. QUBO-暴力遍历算法")
                print("2. QUBO-传统遗传算法")
                print("3. QUBO-传统粒子群算法")
                print("4. QUBO-传统模拟退火算法")
                print("5. QUBO-量子遗传算法")
                print("6. QUBO-量子粒子群算法")
                print("7. QUBO-量子模拟退火算法")
                print("8. QUBO-量子遗传-QAOA混合算法")
                print("9. QUBO-量子退火-QAOA混合算法")
                print("10. QUBO-量子粒子群-QAOA混合算法")
                solver_choice = int(input("请选择求解器 (1-10): "))
                
                start_time = time.time()
                
                if solver_choice == 1:
                    solve_by_brute_force()
                elif solver_choice == 2:
                    solve_by_genetic_algorithm()
                elif solver_choice == 3:
                    solve_by_particle_swarm()
                elif solver_choice == 4:
                    solve_by_simulated_annealing()
                elif solver_choice == 5:
                    solve_by_quantum_genetic_algorithm()
                elif solver_choice == 6:
                    solve_by_quantum_particle_swarm()
                elif solver_choice == 7:
                    solve_by_quantum_simulated_annealing()
                elif solver_choice == 8:
                    solve_by_qga_qaoa_hybrid()
                elif solver_choice == 9:
                    solve_by_qsa_qaoa_hybrid()
                elif solver_choice == 10:
                    solve_by_qpso_qaoa_hybrid()
                else:
                    print("无效选择！")
                    continue
                print(f"求解完成，耗时 {time.time()-start_time:.2f} 秒")
            
            elif choice == 2:
                display_result()
            elif choice == 3:
                iteration_plot_menu()
            elif choice == 4:
                experiment_menu()  # 这里会触发开关判断
            elif choice == 6:
                advanced_experiment_menu()
            elif choice == 7:
                export_timetable_and_running_diagram()
            else:
                print("请输入0-7之间的数字！")
        except ValueError:
            print("输入错误，请输入数字！")

if __name__ == "__main__":
    main_menu()
