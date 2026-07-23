import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ===================== 中文字体配置 =====================
def setup_chinese_font():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in sorted(os.listdir(script_dir)):
        if fname.lower().endswith(('.ttf', '.ttc', '.otf')):
            fpath = os.path.join(script_dir, fname)
            font_manager.fontManager.addfont(fpath)
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

# ===================== 映射表 =====================
OPTIMIZER_NAME_MAP = {
    'sgd': 'SGD',
    'adamk': 'Adam',
}

ATTACK_NAME_MAP = {
    'withoutatt': 'No-Attack',
    'randomnoise': 'Random-Noise',
    'signflipping': 'Sign-Flipping',
    'zerogradient': 'Zero-Gradient',
}

ATTACK_NAME_CN = {
    'No-Attack': '无攻击',
    'Random-Noise': 'RN',
    'Sign-Flipping': 'SF',
    'Zero-Gradient': 'ZG',
}

# ===================== 消融实验配置 =====================
ABLATION_CONFIGS = [
    {
        'name': 'k',
        'data_dir': '/home/lhxu/FL_Compress/results_cifa_1030_adamk_k',
        'aggregate_map': {
            'compress4clipk1': 'k=1',
            'compress4clipk100': 'k=100',
            'compress4clipk300': 'k=300',
            'compress4clipk500': 'k=500',
            'compress4clipk750': 'k=750',
            'compress4clipk1000': 'k=1000',
            'compress4clipk2345': 'k=2345',
        },
        'color_map': {
            'k=1': '#ff7f0e',
            'k=100': '#8c564b',
            'k=300': '#e377c2',
            'k=500': '#000080',
            'k=750': '#d62728',
            'k=1000': '#ff00ff',
            'k=2345': '#008080',
        },
    },
    {
        'name': 'r',
        'data_dir': '/home/lhxu/FL_Compress/results_cifa_1029_adamk_r',
        'aggregate_map': {
            'compress4clipm4691r0.33': 'r=4',
            'compress4clipr1': 'r=1',
            'compress4clipr2': 'r=2',
            'compress4clipr3': 'r=3',
        },
        'color_map': {
            'r=1': '#FFB6C1',
            'r=2': '#DC143C',
            'r=3': '#8B4513',
            'r=4': '#DAA520',
        },
    },
    {
        'name': 'm',
        'data_dir': '/home/lhxu/FL_Compress/results_cifa_1029_adamk_m',
        'aggregate_map': {
            'compress4cliplr6.5e-5': 'm=6',
            'compress4clipm4691r0.33': 'm=4691',
            'compress4clipr0.33': 'm=2382',
        },
        'color_map': {
            'm=6': '#ff00ff',
            'm=4691': '#d62728',
            'm=2382': '#1f77b4',
        },
    },
]

# ===================== 解析目录名 =====================
def parse_dirname(dirname, aggregate_map):
    dirname_lower = dirname.lower()
    optimizer = None
    aggregation = None
    attack = None

    for opt_key in sorted(OPTIMIZER_NAME_MAP.keys(), key=len, reverse=True):
        if opt_key in dirname_lower:
            optimizer = opt_key
            break

    for agg_key in sorted(aggregate_map.keys(), key=len, reverse=True):
        if agg_key in dirname_lower:
            aggregation = agg_key
            break

    for atk_key in sorted(ATTACK_NAME_MAP.keys(), key=len, reverse=True):
        if atk_key in dirname_lower:
            attack = ATTACK_NAME_MAP[atk_key]
            break

    return optimizer, aggregation, attack

# ===================== 读取数据 =====================
def read_acc_data(folder_path):
    pth_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pth')])
    if not pth_files:
        return []

    last_pth = pth_files[-1]
    m = re.match(r'state_round_(\d+)\.pth', last_pth)
    if not m:
        return []

    file_path = os.path.join(folder_path, last_pth)
    try:
        checkpoint = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"[WARN] Load {file_path} failed: {e}")
        return []

    acc_list = [
        a['top1_accuracy'] for a in checkpoint.get('history', {}).get('metrics', [])
        if 'top1_accuracy' in a
    ]

    return acc_list

# ===================== 组织数据 =====================
def organize_data(root_dir, aggregate_map):
    data = {}
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for subdir in subdirs:
        optimizer, aggregation, attack = parse_dirname(subdir, aggregate_map)
        if not optimizer or not aggregation or not attack:
            continue

        opt_name = OPTIMIZER_NAME_MAP[optimizer]
        agg_name = aggregate_map[aggregation]

        acc_list = read_acc_data(os.path.join(root_dir, subdir))
        if not acc_list:
            continue

        data.setdefault(attack, {}).setdefault(agg_name, {})[opt_name] = acc_list
        print(f"[INFO] Loaded: attack={attack}, aggregate={agg_name}, optimizer={opt_name}, epochs={len(acc_list)}")

    return data

# ===================== 绘图 =====================
def plot_ablation(data, config, output_dir, ylabel):
    if not data:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

    aggregate_map = config['aggregate_map']
    color_map = config['color_map']

    max_epochs = 0

    for attack, attack_data in data.items():
        for agg_name in aggregate_map.values():
            if agg_name not in attack_data:
                continue
            for optimizer, acc_list in attack_data[agg_name].items():
                if len(acc_list) > max_epochs:
                    max_epochs = len(acc_list)
                x = np.arange(len(acc_list))
                y = acc_list
                color = color_map.get(agg_name, 'black')
                ax.plot(x, y, color=color, linewidth=2, label=agg_name)

    attack_cn = ATTACK_NAME_CN.get(list(data.keys())[0], list(data.keys())[0])
    ax.set_title(f'攻击: {attack_cn}', fontsize=25)
    ax.set_xlabel('轮数 / 300步', fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_xlim(0, max_epochs)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=22)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"ablation_{config['name']}_acc.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[保存] {save_path}")
    plt.close()


if __name__ == '__main__':
    setup_chinese_font()
    OUTPUT_DIR = '/home/lhxu/FL_Compress/chinese_plots'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for config in ABLATION_CONFIGS:
        print(f"\n{'='*50}")
        print(f"处理消融实验: {config['name']}")
        print(f"{'='*50}")
        data = organize_data(config['data_dir'], config['aggregate_map'])
        if data:
            plot_ablation(data, config, OUTPUT_DIR, '精度')
        else:
            print(f"[跳过] {config['name']} 无数据")
