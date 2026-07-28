import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re

ROOT_DIR = '/data-store/xulihan/FL_compress/results/new_abl_beta1_beta2'

# ---------- parsing ----------

def decode_beta1(enc):
    """Decode b1XX encoding to float. e.g. b100->0.0, b105->0.5, b109->0.9, b1099->0.99"""
    digits = enc[2:]  # strip 'b1'
    return float('0.' + digits.lstrip('0')) if digits != '00' else 0.0

def decode_beta2(enc):
    """Decode b2XXX encoding to float. e.g. b209->0.9, b2099->0.99, b20999->0.999, b209999->0.9999"""
    digits = enc[2:]  # strip 'b2'
    return float('0.' + digits.lstrip('0'))

def parse_dirname(dirname):
    """Parse dirname like cifar10_adamk_compress4_clip_k469_r4_b100_b209_foe_iid"""
    m = re.match(
        r'cifar10_adamk_compress4_clip_k\d+_r\d+_(b1\d+)_(b2\d+)_(\w+)_(iid|noniid)',
        dirname
    )
    if not m:
        return None
    b1_enc, b2_enc, attack, iid_flag = m.groups()
    beta1 = decode_beta1(b1_enc)
    beta2 = decode_beta2(b2_enc)
    return beta1, beta2, attack, iid_flag

# ---------- data loading ----------

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
    """Returns: data[iid_flag][attack][(beta1, beta2)] = acc_list"""
    data = {}
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for subdir in subdirs:
        parsed = parse_dirname(subdir)
        if parsed is None:
            print(f"[WARN] Skipping '{subdir}'")
            continue
        beta1, beta2, attack, iid_flag = parsed
        folder_path = os.path.join(root_dir, subdir)
        acc_list = read_acc_list(folder_path)
        if not acc_list:
            print(f"[WARN] No data in '{subdir}'")
            continue
        data.setdefault(iid_flag, {})
        data[iid_flag].setdefault(attack, {})
        data[iid_flag][attack][(beta1, beta2)] = acc_list
        print(f"[INFO] Loaded: iid={iid_flag}, attack={attack}, beta1={beta1}, beta2={beta2}, "
              f"rounds={len(acc_list)}, max_acc={max(acc_list):.4f}")
    return data

# ---------- display mappings ----------

ATTACK_DISPLAY = {
    'withoutatt': 'No Attack',
    'foe': 'FoE',
    'labelflipping': 'Label Flipping',
    'signflipping': 'Sign Flipping',
}

ATTACK_ORDER = ['withoutatt', 'foe', 'labelflipping', 'signflipping']

# ---------- plotting ----------

def plot_heatmap_figure(data_iid, iid_label, save_path):
    attacks = [a for a in ATTACK_ORDER if a in data_iid]
    if not attacks:
        print(f"[WARN] No attacks found for {iid_label}")
        return

    # Collect all unique beta1 and beta2 values across attacks
    all_beta1 = set()
    all_beta2 = set()
    for attack in attacks:
        for (b1, b2) in data_iid[attack].keys():
            all_beta1.add(b1)
            all_beta2.add(b2)
    beta1_vals = sorted(all_beta1)
    beta2_vals = sorted(all_beta2)

    n_attacks = len(attacks)

    # Each subplot should be a square; set unit size per subplot
    unit = 4.2  # inches per subplot
    fig, axes = plt.subplots(1, n_attacks, figsize=(unit * n_attacks, unit), dpi=300)
    if n_attacks == 1:
        axes = [axes]

    # Collapse-failed runs (max_acc near random ~0.1) are treated separately so
    # they do not distort the colour scale of the converged runs.
    COLLAPSE_THRESH = 0.3  # well below any converged run, well above random 0.1

    # Compute vmin/vmax over only the converged runs (shared across attacks)
    converged_accs = []
    for attack in attacks:
        for acc_list in data_iid[attack].values():
            m = max(acc_list)
            if m >= COLLAPSE_THRESH:
                converged_accs.append(m)
    vmin = min(converged_accs) if converged_accs else 0.0
    vmax = max(converged_accs) if converged_accs else 1.0

    # Sequential single-hue colour map (red) with a bad-colour for collapsed cells
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color='#d9d9d9')  # light gray for collapsed / missing cells

    for ax_idx, attack in enumerate(attacks):
        ax = axes[ax_idx]
        attack_data = data_iid[attack]

        # Build 2-D matrix: rows=beta1, cols=beta2
        mat = np.full((len(beta1_vals), len(beta2_vals)), np.nan)
        collapsed = np.zeros_like(mat, dtype=bool)
        for (b1, b2), acc_list in attack_data.items():
            i = beta1_vals.index(b1)
            j = beta2_vals.index(b2)
            m = max(acc_list)
            if m < COLLAPSE_THRESH:
                collapsed[i, j] = True  # leave as NaN -> bad colour
            else:
                mat[i, j] = m

        im = ax.imshow(mat, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax,
                       interpolation='nearest')

        # Annotate each cell with the accuracy value
        for i in range(len(beta1_vals)):
            for j in range(len(beta2_vals)):
                if collapsed[i, j]:
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=11, fontweight='bold', color='#8a8a8a')
                elif not np.isnan(mat[i, j]):
                    val = mat[i, j]
                    text_color = 'white' if val > (vmin + vmax) / 2 else '#0b0b0b'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=9, fontweight='bold', color=text_color)

        # Axis labels
        ax.set_xticks(range(len(beta2_vals)))
        ax.set_xticklabels([f'{b:.4g}' for b in beta2_vals], fontsize=9)
        ax.set_yticks(range(len(beta1_vals)))
        ax.set_yticklabels([f'{b:.4g}' for b in beta1_vals], fontsize=9)
        ax.set_xlabel(r'$\beta_2$', fontsize=13)
        if ax_idx == 0:
            ax.set_ylabel(r'$\beta_1$', fontsize=13)
        ax.set_title(ATTACK_DISPLAY.get(attack, attack), fontsize=14)

    # No colour bar — values are already annotated in each cell

    plt.suptitle(f'Ablation on $\\beta_1$ / $\\beta_2$ ({iid_label})',
                 fontsize=16, y=1.02)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.88, wspace=0.40)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Figure saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    data = organize_data(ROOT_DIR)

    for iid_flag in ['iid', 'noniid']:
        if iid_flag not in data:
            continue
        iid_label = 'IID' if iid_flag == 'iid' else 'Non-IID'
        save_path = os.path.join(ROOT_DIR, f'abl_beta1_beta2_{iid_flag}.png')
        plot_heatmap_figure(data[iid_flag], iid_label, save_path)
