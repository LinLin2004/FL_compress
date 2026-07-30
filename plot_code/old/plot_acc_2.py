import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import csv

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
    # 'compress4cliplr6.5e-5':'Compress+MultiKrum_p_clip_lr6.5e-5',
    # 'compress4cliplr7e-5':'Compress+MultiKrum_p_clip_lr7e-5',
    # 'compress4cliplr8e-5':'Compress+MultiKrum_p_clip_lr8e-5',
    # 'compress4cliplr9e-5':'Compress+MultiKrum_p_clip_lr9e-5'
    # 'compress4p0.2':'Compress+MultiKrum_p_0.2',
    # 'compress4beta20.9':'Compress+MultiKrum_beta2_0.9',
    # 'compress4beta20.88':'Compress+MultiKrum_beta2_0.88'
    # 'compress4clipm4691r0.33':'Compress+MultiKrum_m4691_r0.33',
    # 'compress4clipr0.33':'Compress+MultiKrum_m2382_r0.33',
    
}
OPTIMIZER_NAME_MAP = {
    'sgd': 'SGD',
    'adamk': 'AdamK'
}

ITERS_PER_EPOCH = {
    'SGD': 300,
    'AdamK': 300
}

def capitalize_attack(s: str) -> str:
    return ''.join(word.capitalize() for word in s.split('_'))

def get_info_from_dirname(dirname):
    parts = dirname.split('_')
    optimizer = None
    aggregation = None
    attack = None
    ignored_words = {'ijcnn1', 'covtype', 'covtype_nesterov', 'cif', 'cifa', 'katyusha','CIF'}
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
            acc_list = history.get('metrics', [])
            acc_list = [a['top1_accuracy'] for a in acc_list]
            if not acc_list:
                continue
            rounds_data.append((round_num, acc_list))
    rounds_data.sort(key=lambda x: x[0])
    return rounds_data

def organize_data(root_dir):
    data = {}
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"[INFO] Found {len(subdirs)} subdirectories in {root_dir}")
    for subdir in subdirs:
        optimizer_raw, aggregate_raw, attack_raw = get_info_from_dirname(subdir)
        if optimizer_raw is None or aggregate_raw is None or attack_raw is None:
            print(f"[WARN] Skipping '{subdir}' due to parsing failure")
            continue
        optimizer = OPTIMIZER_NAME_MAP.get(optimizer_raw, optimizer_raw.capitalize())
        aggregate = AGGREGATE_NAME_MAP.get(aggregate_raw, aggregate_raw.capitalize())
        attack = attack_raw
        folder_path = os.path.join(root_dir, subdir)
        rounds_data = read_round_pth_files(folder_path)
        if len(rounds_data) == 0:
            print(f"[WARN] No valid rounds data in folder: {folder_path}")
            continue
        round_num, loss_list = rounds_data[-1]
        data.setdefault(attack, {})
        data[attack].setdefault(aggregate, {})
        data[attack][aggregate][optimizer] = (round_num, loss_list)
        print(f"[INFO] Loaded: attack={attack}, aggregate={aggregate}, optimizer={optimizer}, rounds={round_num}")
    return data

def sample_loss_for_10_points(loss_list, iters_per_epoch):
    total_epochs = len(loss_list)
    total_iters = iters_per_epoch * total_epochs
    sample_iters = np.linspace(0, total_iters, 10)
    sample_epochs = np.clip((sample_iters / iters_per_epoch).astype(int), 0, total_epochs - 1)
    sampled_loss = [loss_list[i] for i in sample_epochs]
    return sample_iters, sampled_loss

def plot_data_with_10_points(data, save_path=None):
    attack_order = ['Withoutatt', 'Randomnoise', 'Signflipping', 'Zerogradient']
    attack_order = [capitalize_attack(name) for name in attack_order]
    attacks_present = [atk for atk in attack_order if atk in data]
    if not attacks_present:
        print("[WARN] No matching attack data found. Skipping plot.")
        return
    
    aggregates = list(AGGREGATE_NAME_MAP.values())
    optimizers = list(OPTIMIZER_NAME_MAP.values())
    
    # 为每个聚合方法指定颜色
    aggregate_color_map = {
        'GeoMed': 'orange',
        'CwMed': 'green',
        'Mean': 'blue',
        'Krum': 'purple',
        'Compress+MultiKrum': 'red',
        'Compress+FABA': 'pink',
        'Compress+MultiKrum_p_clip': 'brown',
        # 'Compress+MultiKrum_p_clip_lr1e-4':'blue',
        # 'Compress+MultiKrum_p_clip_lr4.5e-5':'green',
        # 'Compress+MultiKrum_p_clip_lr5e-5':'yellow',
        # 'Compress+MultiKrum_p_clip_lr5.5e-5':'pink',
        # 'Compress+MultiKrum_p_clip_lr6e-5':'cyan',
        # 'Compress+MultiKrum_p_clip_lr6.5e-5':'magenta',
        # 'Compress+MultiKrum_p_clip_lr7e-5':'orange',
        # 'Compress+MultiKrum_p_clip_lr8e-5':'purple',
        # 'Compress+MultiKrum_p_clip_lr9e-5':'black',
        # 'Compress+MultiKrum_p_0.2': 'pink',
        # 'Compress+MultiKrum_beta2_0.9': 'yellow',
        # 'Compress+MultiKrum_beta2_0.88': 'brown',
        # 'Compress+MultiKrum_m4691_r0.33': 'red',
        # 'Compress+MultiKrum_m2382_r0.33': 'blue',
    }
    
    line_styles = {'AdamK': '-', 'SGD': '--', 'Nesterov': '-', 'SAGA': '-.', 'SVRG': ':'}
    # marker_map = {'Katyusha': '^', 'SGD': 'd', 'Nesterov': 's', 'SAGA': 'o', 'SVRG': 's'}
    marker_map = {'AdamK': '^', 'SGD': '^', 'Nesterov': '^', 'SAGA': '^', 'SVRG': '^'}
    
    num_attacks = len(attacks_present)
    rows = math.ceil(num_attacks / 4)
    cols = min(4, num_attacks)
    print(f"[INFO] Plotting {num_attacks} attacks, grid size {rows}x{cols}")
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), dpi=300)
    if rows * cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for ax_idx, attack in enumerate(attacks_present):
        ax = axes[ax_idx]
        attack_data = data[attack]
        max_epoch_len = 0
        for aggregate in aggregates:
            if aggregate not in attack_data:
                continue
            for optimizer in optimizers:
                if optimizer not in attack_data[aggregate]:
                    continue
                round_num, loss_list = attack_data[aggregate][optimizer]
                iters_per_epoch = ITERS_PER_EPOCH.get(optimizer, 1000)
                total_iter = iters_per_epoch * (len(loss_list) - 1)
                xvals, yvals = sample_loss_for_10_points(loss_list, iters_per_epoch)
                xvals = np.arange(len(yvals))
                
                # Special case highlight in red
                if aggregate == 'Compress+MultiKrum' and optimizer == 'AdamK':
                    color = 'red'  # 强调Krum + Katyusha为红色
                else:
                    color = aggregate_color_map.get(aggregate, 'black')  # Use aggregation's color

                linestyle = line_styles.get(optimizer, '-')
                marker = marker_map.get(optimizer, None)
                linewidth = 1.2
                label = f"{aggregate} + {optimizer}"
                ax.plot(xvals, yvals, linestyle=linestyle, color=color, label=label, linewidth=linewidth, marker=marker, markersize=7)
                if len(yvals) > max_epoch_len:
                    max_epoch_len = len(yvals)
        ax.set_title(f'Attack: {attack}', fontsize=25)
        ax.set_xlabel('k iteration / 3000', fontsize=25)
        ax.set_ylabel('Accuracy', fontsize=23)
        ax.set_xlim(0, 10)
        ticks = np.arange(10)
        ticklabels = [str(t) for t in ticks]
        # print(ticklabels)
        # ticklabels[-1] = '10'
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=20)
        ax.tick_params(axis='y', labelsize=13)
        # ax.set_ylim(0.6, 2.60)  # 修改后的 y 轴范围
        ax.grid(False)
        ax.legend(fontsize='large')
    
    for i in range(num_attacks, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] Figure saved to {save_path}")
    plt.show()

def save_best_acc_to_csv(data, save_csv_path):
    rows = []
    for attack, aggs in data.items():
        for aggregate, opts in aggs.items():
            for optimizer, (round_num, acc_list) in opts.items():
                best_acc = max(acc_list) if acc_list else None
                rows.append([attack, aggregate, optimizer, best_acc])
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    with open(save_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Attack', 'Aggregate', 'Optimizer', 'Best_Acc'])
        writer.writerows(rows)
    print(f"[INFO] Saved best acc CSV to {save_csv_path}")

if __name__ == '__main__':
    ROOT_DIR = '/home/lhxu/FL_Compress/results_cifa_1110_sgd'  # 替换为实际路径
    # ITERS_PER_EPOCH = 1000
    print("[INFO] Organizing data from", ROOT_DIR)
    data = organize_data(ROOT_DIR)
    print("[INFO] Loaded attack types:", list(data.keys()))
    csv_path = os.path.join(ROOT_DIR, 'CIF_comparsion_best_acc.csv')
    save_best_acc_to_csv(data, csv_path)
    plot_save_path = os.path.join(ROOT_DIR, 'CIF_acc_plot_10points.png')
    plot_data_with_10_points(data, save_path=plot_save_path)
