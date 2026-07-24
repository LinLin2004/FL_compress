import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re

ROOT_DIR = '/data-store/xulihan/FL_compress/results/batch_size'

# Parse dirname like cifar10_adamk_compress4_clip_k469_r2_bs32_withoutatt_iid
# or cifar10_adamk_compress4_clip_k469_r2_bs32_withoutatt_mildnoniid_alpha05
# or cifar10_adamk_compress4_clip_k469_r2_bs32_withoutatt_strongnoniid_alpha01
def parse_dirname(dirname):
    m = re.match(
        r'cifar10_adamk_compress4_clip_k\d+_r(\d+)_bs(\d+)_(\w+)_(iid|mildnoniid_alpha\d+|strongnoniid_alpha\d+)',
        dirname
    )
    if not m:
        return None
    r_val = m.group(1)
    bs_val = m.group(2)
    attack = m.group(3)
    iid_flag = m.group(4)
    return bs_val, r_val, attack, iid_flag

def read_acc_list(folder_path):
    pth_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pth')])
    if not pth_files:
        return []
    f = pth_files[-1]
    m = re.match(r'state_round_(\d+).pth', f)
    if not m:
        return []
    file_path = os.path.join(folder_path, f)
    try:
        ckpt = torch.load(file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"[WARN] Load {file_path} failed: {e}")
        return []
    history = ckpt.get('history', {})
    acc_list = history.get('metrics', [])
    return [a['top1_accuracy'] for a in acc_list]

def organize_data(root_dir):
    """Returns: data[iid_flag][attack][bs_val] = acc_list"""
    data = {}
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for subdir in subdirs:
        parsed = parse_dirname(subdir)
        if parsed is None:
            print(f"[WARN] Skipping '{subdir}'")
            continue
        bs_val, r_val, attack, iid_flag = parsed
        folder_path = os.path.join(root_dir, subdir)
        acc_list = read_acc_list(folder_path)
        if not acc_list:
            print(f"[WARN] No data in '{subdir}'")
            continue
        data.setdefault(iid_flag, {})
        data[iid_flag].setdefault(attack, {})
        data[iid_flag][attack][bs_val] = acc_list
        print(f"[INFO] Loaded: iid={iid_flag}, attack={attack}, bs={bs_val}, r={r_val}, rounds={len(acc_list)}")
    return data

IID_DISPLAY = {
    'iid': 'IID',
    'mildnoniid_alpha05': r'Mild Non-IID ($\alpha$=0.5)',
    'strongnoniid_alpha01': r'Strong Non-IID ($\alpha$=0.1)',
}

IID_ORDER = ['iid', 'mildnoniid_alpha05', 'strongnoniid_alpha01']

ATTACK_DISPLAY = {
    'withoutatt': 'No Attack',
    'foe': 'FoE',
    'labelflipping': 'Label Flipping',
    'signflipping': 'Sign Flipping',
}

ATTACK_ORDER = ['withoutatt', 'foe', 'labelflipping', 'signflipping']

# Sort batch size values numerically
def sort_bs(bs_list):
    return sorted(bs_list, key=lambda x: int(x))

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

def plot_figure(data_iid, iid_label, save_path):
    attacks = [a for a in ATTACK_ORDER if a in data_iid]
    if not attacks:
        print(f"[WARN] No attacks found for {iid_label}")
        return

    all_bs = set()
    for attack in attacks:
        all_bs.update(data_iid[attack].keys())
    all_bs = sort_bs(list(all_bs))

    num_attacks = len(attacks)
    fig, axes = plt.subplots(1, num_attacks, figsize=(6 * num_attacks, 5), dpi=300)
    if num_attacks == 1:
        axes = [axes]

    for ax_idx, attack in enumerate(attacks):
        ax = axes[ax_idx]
        attack_data = data_iid[attack]

        for i, bs_val in enumerate(all_bs):
            if bs_val not in attack_data:
                continue
            acc_list = attack_data[bs_val]
            xvals = np.arange(len(acc_list))
            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            ax.plot(xvals, acc_list, color=color, label=f'BS={bs_val}', linewidth=1.5)
            step = max(1, len(acc_list) // 10)
            sample_idx = np.arange(0, len(acc_list), step)
            ax.plot(sample_idx, [acc_list[j] for j in sample_idx], color=color,
                    marker=marker, markersize=6, linestyle='None')

        ax.set_title(f'{ATTACK_DISPLAY.get(attack, attack)}', fontsize=18)
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='lower right')

    plt.suptitle(f'Ablation on Batch Size ({iid_label})', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Figure saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    data = organize_data(ROOT_DIR)

    for iid_flag in IID_ORDER:
        if iid_flag not in data:
            continue
        iid_label = IID_DISPLAY.get(iid_flag, iid_flag)
        save_path = os.path.join(ROOT_DIR, f'abl_bs_{iid_flag}.png')
        plot_figure(data[iid_flag], iid_label, save_path)
