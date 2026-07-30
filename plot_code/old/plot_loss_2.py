import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import csv

# ================================
# 攻击名称统一映射（你要求的格式）
# ================================
ATTACK_NAME_MAP = {
    'withoutatt': 'No-Attack',
    'randomnoise': 'Random-Noise',
    'signflipping': 'Sign-Flipping',
    'zerogradient': 'Zero-Gradient',
}

AGGREGATE_NAME_MAP = {
    'geometric': 'GeoMed',
    'median': 'CwMed',
    'mean': 'Mean',
    'krum': 'Krum',
    'compress4_clip': 'FedAdam-BACK',
    'compress4_clip':'FedAdam-BACK',
    # 'compress4cliplr1e-4':'Compress+MultiKrum_p_clip_lr1e-4',
    # 'compress4cliplr4.5e-5':'Compress+MultiKrum_p_clip_lr4.5e-5',
    # 'compress4cliplr5e-5':'Compress+MultiKrum_p_clip_lr5e-5',
    # 'compress4cliplr5.5e-5':'Compress+MultiKrum_p_clip_lr5.5e-5',
    # 'compress4cliplr6e-5':'Compress+MultiKrum_p_clip_lr6e-5',
    # 'compress4cliplr6.5e-5':'m=6',
    # 'compress4cliplr7e-5':'Compress+MultiKrum_p_clip_lr7e-5',
    # 'compress4cliplr8e-5':'Compress+MultiKrum_p_clip_lr8e-5',
    # 'compress4cliplr9e-5':'Compress+MultiKrum_p_clip_lr9e-5'
    # 'compress4p0.2':'Compress+MultiKrum_p_0.2',
    # 'compress4beta20.9':'Compress+MultiKrum_beta2_0.9',
    # 'compress4beta20.88':'Compress+MultiKrum_beta2_0.88'
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

def format_attack_name(raw: str):
    raw = raw.lower()
    if raw in ATTACK_NAME_MAP:
        return ATTACK_NAME_MAP[raw]
    return raw.capitalize()

def get_info_from_dirname(dirname):
    parts = dirname.split('_')
    optimizer = None
    aggregation = None
    attack = None
    ignored_words = {'ijcnn1', 'covtype', 'covtype_nesterov', 'cif', 'cifa', 'katyusha', 'CIF'}

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
                attack = format_attack_name(p_low)

    print(f"[DEBUG] Parsing '{dirname}': optimizer={optimizer}, aggregation={aggregation}, attack={attack}")
    return optimizer, aggregation, attack

def read_round_pth_files(folder_path):
    pth_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pth')])
    rounds_data = []

    for f in pth_files[-1:]:
        m = re.match(r'state_round_(\d+).pth', f)
        if m:
            round_num = int(m.group(1))
            file_path = os.path.join(folder_path, f)

            try:
                checkpoint = torch.load(file_path, map_location='cpu')
            except Exception as e:
                print(f"[WARN] Load {file_path} failed: {e}")
                continue

            history = checkpoint.get('history', {})
            loss_list = history.get('loss', [])
            if not loss_list:
                continue

            rounds_data.append((round_num, loss_list))

    return rounds_data

def organize_data(root_dir):
    data = {}
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    print(f"[INFO] Found {len(subdirs)} subdirectories in {root_dir}")

    for subdir in subdirs:
        optimizer_raw, aggregate_raw, attack = get_info_from_dirname(subdir)

        if optimizer_raw is None or aggregate_raw is None or attack is None:
            print(f"[WARN] Skipping '{subdir}' due to parsing failure")
            continue

        optimizer = OPTIMIZER_NAME_MAP[optimizer_raw]
        aggregate = AGGREGATE_NAME_MAP[aggregate_raw]

        rounds_data = read_round_pth_files(os.path.join(root_dir, subdir))
        if not rounds_data:
            continue

        round_num, acc_list = rounds_data[0]

        data.setdefault(attack, {}).setdefault(aggregate, {})[optimizer] = (round_num, acc_list)

        print(f"[INFO] Loaded: attack={attack}, aggregate={aggregate}, optimizer={optimizer}, epochs={len(acc_list)}")

    return data

def plot_full_curves_with_markers(data, save_path=None):
    attack_order = ['No-Attack', 'Random-Noise', 'Sign-Flipping', 'Zero-Gradient']
    attacks_present = [atk for atk in attack_order if atk in data]

    aggregates = list(AGGREGATE_NAME_MAP.values())
    optimizers = list(OPTIMIZER_NAME_MAP.values())

    aggregate_color_map = {
        'GeoMed': 'orange',
        'CwMed': 'green',
        'Mean': 'blue',
        'Krum': 'purple',
        'FedAdam-BACK': 'red',
        # 'm=6':'magenta',
        # 'm=4691': 'red',
        # 'm=2382': 'blue',
        # 'r=1': 'pink',
        # 'r=2': 'red',
        # 'r=3': 'brown',
        # 'r=4': 'orange',
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

    num_attacks = len(attacks_present)
    rows = math.ceil(num_attacks / 4)
    cols = min(4, num_attacks)

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), dpi=300)
    if rows * cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for ax_idx, attack in enumerate(attacks_present):
        ax = axes[ax_idx]
        attack_data = data[attack]
        max_epochs = 0

        for aggregate in aggregates:
            if aggregate not in attack_data:
                continue

            for optimizer in optimizers:
                if optimizer not in attack_data[aggregate]:
                    continue

                _, acc_list = attack_data[aggregate][optimizer]

                if len(acc_list) > max_epochs:
                    max_epochs = len(acc_list)

                x = np.arange(len(acc_list))
                y = acc_list

                ax.plot(
                    x,
                    y,
                    linestyle=line_styles[optimizer],
                    color=aggregate_color_map.get(aggregate, 'black'),
                    linewidth=1.5,
                    label=f"{aggregate}"
                )

        ax.set_title(f'Attack: {attack}', fontsize=25)
        ax.set_xlabel('k steps / 300', fontsize=25)
        ax.set_ylabel('Loss', fontsize=25)
        ax.set_xlim(0, max_epochs)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize='large')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] Figure saved to {save_path}")
    plt.show()

def save_best_acc_to_csv(data, save_csv_path):
    rows = []
    for attack, aggs in data.items():
        for aggregate, opts in aggs.items():
            for optimizer, (_, acc_list) in opts.items():
                rows.append([attack, aggregate, optimizer, max(acc_list)])

    with open(save_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Attack', 'Aggregate', 'Optimizer', 'Best_Acc'])
        writer.writerows(rows)

if __name__ == "__main__":
    ROOT_DIR = "/home/lhxu/FL_Compress/results_cifa_1030_adamk_k"

    data = organize_data(ROOT_DIR)

    save_best_acc_to_csv(data, os.path.join(ROOT_DIR, "CIF_comparsion_best_loss.csv"))

    plot_full_curves_with_markers(
        data,
        save_path=os.path.join(ROOT_DIR, "CIF_loss_plot_full_curve_1000.pdf")
    )
