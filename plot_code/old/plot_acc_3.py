import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import csv

# =============== 映射表 ===============

AGGREGATE_NAME_MAP = {
    'geometric': 'GeoMed',
    'median': 'CwMed',
    'mean': 'Mean',
    'krum': 'Krum',
    'compress2':'Compress+FABA',
    'compress4':'Compress+MultiKrum',
    'compress4clip':'Compress+MultiKrum_p_clip',
    # 'compress4cliplr1e-4':'Compress+MultiKrum_p_clip_lr1e-4', 
    # 'compress4cliplr4.5e-5':'Compress+MultiKrum_p_clip_lr4.5e-5', 
    # 'compress4cliplr5e-5':'Compress+MultiKrum_p_clip_lr5e-5', 
    # 'compress4cliplr5.5e-5':'Compress+MultiKrum_p_clip_lr5.5e-5', 
    # 'compress4cliplr6e-5':'Compress+MultiKrum_p_clip_lr6e-5', 
    # 'compress4cliplr6.5e-5':'m=6', 
    # 'compress4cliplr7e-5':'Compress+MultiKrum_p_clip_lr7e-5', 
    # 'compress4cliplr8e-5':'Compress+MultiKrum_p_clip_lr8e-5', 
    # 'compress4cliplr9e-5':'Compress+MultiKrum_p_clip_lr9e-5', 
    # 'compress4p0.2':'Compress+MultiKrum_p_0.2', 
    # 'compress4beta20.9':'Compress+MultiKrum_beta2_0.9', 
    # 'compress4beta20.88':'Compress+MultiKrum_beta2_0.88', 
    # 'compress4clipm4691r0.33':'m=4691', 
    # 'compress4clipr0.33':'m=2382',
    # 'compress4clipm4691r0.33':'r=4',
    # 'compress4clipr1':'r=1',
    # 'compress4clipr2':'r=2',
    # 'compress4clipr3':'r=3',
    'compress4clipk1':'k=1', 
    'compress4clipk100':'k=100', 
    'compress4clipk300':'k=300', 
    'compress4clipk500':'k=500', 
    'compress4clipk750':'k=750', 
    'compress4clipk1000':'k=1000',
    'compress4clipk2345':'k=2345',
}

OPTIMIZER_NAME_MAP = {
    'sgd': 'SGD',
    'adamk': 'AdamK'
}

# ★★★★★ 你的攻击名最终映射（论文格式） ★★★★★
ATTACK_NAME_MAP = {
    'Withoutatt': 'No-Attack',
    'Randomnoise': 'Random-Noise',
    'Signflipping': 'Sign-Flipping',
    'Zerogradient': 'Zero-Gradient',
}


# =============== 解析函数 ===============

def capitalize_attack(s: str) -> str:
    return ''.join(word.capitalize() for word in s.split('_'))

def get_info_from_dirname(dirname):
    parts = dirname.split('_')
    optimizer = None
    aggregation = None
    attack = None
    ignored_words = {
        'ijcnn1','covtype','covtype_nesterov',
        'cif','cifa','katyusha','CIF'
    }

    for p in parts:
        p_low = p.lower()
        if p_low in OPTIMIZER_NAME_MAP:
            optimizer = p_low
        elif p_low in AGGREGATE_NAME_MAP:
            aggregation = p_low
        elif p_low in ignored_words:
            continue
        elif re.match(r'^\d+(\.\d+)?$', p_low):
            continue
        else:
            if attack is None:
                attack = p_low

    if attack:
        attack = capitalize_attack(attack)

    print(f"[DEBUG] Parsing '{dirname}': optimizer={optimizer}, aggregation={aggregation}, attack={attack}")
    return optimizer, aggregation, attack


# =============== 加载 pth ===============

def read_round_pth_files(folder_path):
    pth_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pth')])
    if not pth_files:
        return []

    last_pth = pth_files[-1]
    m = re.match(r'state_round_(\d+).pth', last_pth)
    if not m:
        return []

    round_num = int(m.group(1))
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

    if not acc_list:
        return []

    return [(round_num, acc_list)]


# =============== 组织数据 ===============

def organize_data(root_dir):
    data = {}
    subdirs = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d))]
    print(f"[INFO] Found {len(subdirs)} subdirectories")

    for subdir in subdirs:
        optimizer_raw, aggregate_raw, attack_raw = get_info_from_dirname(subdir)

        if not optimizer_raw or not aggregate_raw or not attack_raw:
            print(f"[WARN] Skip '{subdir}'")
            continue

        optimizer = OPTIMIZER_NAME_MAP.get(optimizer_raw)
        aggregate = AGGREGATE_NAME_MAP.get(aggregate_raw)

        # ★★★★★ HERE: 强制使用你的攻击映射 ★★★★★
        attack = ATTACK_NAME_MAP.get(attack_raw, attack_raw)

        folder_path = os.path.join(root_dir, subdir)
        rounds_data = read_round_pth_files(folder_path)
        if not rounds_data:
            continue

        round_num, acc_list = rounds_data[0]

        data.setdefault(attack, {})
        data[attack].setdefault(aggregate, {})
        data[attack][aggregate][optimizer] = (round_num, acc_list)

        print(f"[INFO] Loaded: {attack}, {aggregate}, {optimizer}, epochs={len(acc_list)}")

    return data


# =============== 绘图 ===============

def plot_full_curves_with_markers(data, save_path=None):

    # ★★★★★ 使用你论文中的攻击顺序 ★★★★★
    attack_order = [
        'No-Attack',
        'Random-Noise',
        'Sign-Flipping',
        'Zero-Gradient'
    ]

    attacks_present = [a for a in attack_order if a in data]

    fig, axes = plt.subplots(
        nrows=1, ncols=len(attacks_present),
        figsize=(7 * len(attacks_present), 6),
        dpi=300
    )

    if len(attacks_present) == 1:
        axes = [axes]

    aggregate_color_map = {
        'GeoMed': 'orange',
        'CwMed': 'green',
        'Mean': 'blue',
        'Krum': 'purple',
        'FedAdam-BACK': 'red',
        'FedAdam-BACK_p_clip': 'brown',
        'm=6':'magenta', 
        'm=4691': 'red', 
        'm=2382': 'blue',
        'r=1': 'pink',
        'r=2': 'red',
        'r=3': 'brown',
        'r=4': 'orange',
        'k=1': 'orange', 
        'k=100': 'brown', 
        'k=300': 'pink', 
        'k=500': 'navy',
        'k=750': 'red', 
        'k=1000': 'magenta', 
        'k=1500': 'purple',
        'k=2345': 'teal',
    }

    line_styles = {'AdamK': '-', 'SGD': '--'}

    for ax, attack in zip(axes, attacks_present):

        attack_data = data[attack]
        max_epochs = 0

        for aggregate in AGGREGATE_NAME_MAP.values():
            if aggregate not in attack_data:
                continue
            for optimizer in OPTIMIZER_NAME_MAP.values():
                if optimizer not in attack_data[aggregate]:
                    continue

                _, acc_list = attack_data[aggregate][optimizer]

                if len(acc_list) > max_epochs:
                    max_epochs = len(acc_list)

                x_full = np.arange(len(acc_list))

                color = aggregate_color_map.get(aggregate, 'black')
                linestyle = line_styles.get(optimizer, '-')

                ax.plot(
                    x_full, acc_list,
                    linestyle=linestyle,
                    color=color,
                    linewidth=2,
                    label=f"{aggregate}" #-{optimizer}"
                )

        ax.set_title(f'Attack: {attack}', fontsize=25)
        ax.set_xlabel('k steps / 300', fontsize=25)
        ax.set_ylabel("Accuracy", fontsize=25)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] Saved PDF to {save_path}")

    plt.show()


# =============== CSV ===============

def save_best_acc_to_csv(data, save_csv_path):
    rows = []
    for attack, aggs in data.items():
        for aggregate, opts in aggs.items():
            for optimizer, (_, acc_list) in opts.items():
                rows.append([attack, aggregate, optimizer, max(acc_list)])

    rows.sort()
    with open(save_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Attack", "Aggregate", "Optimizer", "Best_Acc"])
        writer.writerows(rows)

    print(f"[INFO] Saved CSV to {save_csv_path}")


# =============== MAIN ===============

if __name__ == '__main__':
    ROOT_DIR = '/home/lhxu/FL_Compress/results_cifa_1030_adamk_k'

    data = organize_data(ROOT_DIR)
    save_best_acc_to_csv(data, os.path.join(ROOT_DIR, "CIF_best_acc.csv"))

    # ★★★★★ 保存为 PDF ★★★★★
    plot_full_curves_with_markers(
        data,
        save_path=os.path.join(ROOT_DIR, "CIF_acc_plot_full_curve_1000.pdf")
    )
