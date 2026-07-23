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
AGGREGATE_NAME_MAP = {
    'geometric': 'GeoMed',
    'median': 'CwMed',
    'mean': 'Mean',
    'krum': 'Krum',
    'compress4clip': 'CompressedKrum',
}

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

AGGREGATE_COLOR_MAP = {
    'GeoMed': '#1f77b4',
    'CwMed': '#2ca02c',
    'Mean': '#ff7f0e',
    'Krum': '#9467bd',
    'CompressedKrum': '#d62728',
}

LINE_STYLES = {'Adam': '-', 'SGD': '--'}

# ===================== 解析目录名 =====================
def parse_dirname(dirname):
    dirname_lower = dirname.lower()
    optimizer = None
    aggregation = None
    attack = None

    for opt_key in sorted(OPTIMIZER_NAME_MAP.keys(), key=len, reverse=True):
        if opt_key in dirname_lower:
            optimizer = opt_key
            break

    for agg_key in sorted(AGGREGATE_NAME_MAP.keys(), key=len, reverse=True):
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
def organize_data(root_dir):
    data = {}
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for subdir in subdirs:
        optimizer, aggregation, attack = parse_dirname(subdir)
        if not optimizer or not aggregation or not attack:
            continue

        opt_name = OPTIMIZER_NAME_MAP[optimizer]
        agg_name = AGGREGATE_NAME_MAP[aggregation]

        acc_list = read_acc_data(os.path.join(root_dir, subdir))
        if not acc_list:
            continue

        data.setdefault(attack, {}).setdefault(agg_name, {})[opt_name] = acc_list
        print(f"[INFO] Loaded: attack={attack}, aggregate={agg_name}, optimizer={opt_name}, epochs={len(acc_list)}")

    return data

# ===================== 柔光效果 =====================
def add_soft_glow(ax, x, y, color, linestyle):
    for lw, alpha in [(6, 0.12), (10, 0.09), (14, 0.06)]:
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2)

# ===================== 绘图 =====================
def plot_comparison(data, output_dir, basename, ylabel):
    attack_order = ['No-Attack', 'Random-Noise', 'Sign-Flipping', 'Zero-Gradient']
    attacks_present = [a for a in attack_order if a in data]

    if not attacks_present:
        print("[跳过] 没有可用的攻击数据")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 分成 2+2 两组
    groups = [attacks_present[:2], attacks_present[2:]]

    for group_idx, group_attacks in enumerate(groups):
        fig, axes = plt.subplots(1, len(group_attacks), figsize=(7 * len(group_attacks), 6), dpi=300)
        if len(group_attacks) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, attack in zip(axes, group_attacks):
            attack_data = data[attack]
            max_epochs = 0

            for aggregate in AGGREGATE_NAME_MAP.values():
                if aggregate not in attack_data:
                    continue
                for optimizer in OPTIMIZER_NAME_MAP.values():
                    if optimizer not in attack_data[aggregate]:
                        continue

                    acc_list = attack_data[aggregate][optimizer]
                    if len(acc_list) > max_epochs:
                        max_epochs = len(acc_list)

                    x = np.arange(len(acc_list))
                    y = acc_list

                    color = AGGREGATE_COLOR_MAP.get(aggregate, 'black')
                    linestyle = LINE_STYLES.get(optimizer, '-')

                    add_soft_glow(ax, x, y, color, linestyle)

            attack_cn = ATTACK_NAME_CN.get(attack, attack)
            ax.set_title(f'攻击: {attack_cn}', fontsize=25)
            ax.set_xlabel('轮数 / 300步', fontsize=25)
            ax.set_ylabel(ylabel, fontsize=25)
            ax.set_xlim(0, max_epochs)
            ax.grid(True, linestyle='--', alpha=0.3)

            # 自定义 legend
            handles = []
            for aggregate in AGGREGATE_NAME_MAP.values():
                if aggregate not in attack_data:
                    continue
                for optimizer in OPTIMIZER_NAME_MAP.values():
                    if optimizer not in attack_data[aggregate]:
                        continue
                    color = AGGREGATE_COLOR_MAP.get(aggregate, 'black')
                    linestyle = LINE_STYLES.get(optimizer, '-')
                    label = f"{aggregate} ({optimizer})"
                    handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle,
                                              linewidth=2, label=label))
            ax.legend(handles=handles, fontsize='large')

        plt.tight_layout()

        suffix = chr(ord('a') + group_idx)
        filename = f"{basename}_{suffix}.pdf"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[保存] {save_path}")
        plt.close()


if __name__ == '__main__':
    setup_chinese_font()

    ROOT_DIR = '/home/lhxu/FL_Compress/results_main_experiments'
    OUTPUT_DIR = '/home/lhxu/FL_Compress/chinese_plots'

    data = organize_data(ROOT_DIR)
    plot_comparison(data, OUTPUT_DIR, 'comparison_acc', '精度')
