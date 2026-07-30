import os
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
import re

ROOT_DIR = '/data-store/xulihan/FL_compress/results/new_abl_k'

# Parse k value from dirname like cifar10_adamk_compress4_clip_k5_withoutatt_iid
def parse_dirname(dirname):
    # pattern: ..._k{K}_{attack}_{iid|noniid}
    m = re.match(r'cifar10_adamk_compress4_clip_k(\d+)_(\w+)_(iid|noniid)', dirname)
    if not m:
        return None
    k_val = m.group(1)
    attack = m.group(2)
    iid_flag = m.group(3)
    return k_val, attack, iid_flag

def read_acc_list(folder_path):
    pth_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.pth')],
        key=_checkpoint_sort_key,
    )
    if not pth_files:
        return []
    # Use the last checkpoint (final round)
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
    """Returns: data[iid_flag][attack][k_val] = acc_list"""
    data = {}
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for subdir in subdirs:
        parsed = parse_dirname(subdir)
        if parsed is None:
            print(f"[WARN] Skipping '{subdir}'")
            continue
        k_val, attack, iid_flag = parsed
        folder_path = os.path.join(root_dir, subdir)
        acc_list = read_acc_list(folder_path)
        if not acc_list:
            print(f"[WARN] No data in '{subdir}'")
            continue
        data.setdefault(iid_flag, {})
        data[iid_flag].setdefault(attack, {})
        data[iid_flag][attack][k_val] = acc_list
        print(f"[INFO] Loaded: iid={iid_flag}, attack={attack}, k={k_val}, epochs={len(acc_list)}")
    return data

ATTACK_DISPLAY = {
    'withoutatt': 'No Attack',
    'foe': 'FoE',
    'labelflipping': 'Label Flipping',
    'signflipping': 'Sign Flipping',
}

ATTACK_ORDER = ['withoutatt', 'foe', 'labelflipping', 'signflipping']

# Reference-image palette: dark blue -> purple -> magenta -> red/orange -> gold -> yellow.
REFERENCE_PALETTE = [
    '#0d0887',
    '#6a00a8',
    '#b12a90',
    '#cf3f73',
    '#f06b44',
    '#f2b447',
    '#f9f74a',
]

K_COLOR_OVERRIDES = {
    '469': '#e53935',
}

# Sort k values numerically
def sort_k(k_list):
    return sorted(k_list, key=lambda x: int(x))

def build_k_colors(all_k):
    if len(all_k) <= 1:
        color_indices = [0]
    else:
        color_indices = np.linspace(0, len(REFERENCE_PALETTE) - 1, len(all_k))
        color_indices = np.rint(color_indices).astype(int)

    k_to_color = {
        k_val: REFERENCE_PALETTE[color_indices[i]]
        for i, k_val in enumerate(all_k)
    }
    k_to_color.update({
        k_val: color
        for k_val, color in K_COLOR_OVERRIDES.items()
        if k_val in k_to_color
    })
    return k_to_color

# Marker maps for different k values
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

def _checkpoint_sort_key(filename):
    m = re.match(r'state_round_(\d+).pth', filename)
    return int(m.group(1)) if m else -1

def _style_3d_axis(ax):
    ax.computed_zorder = False
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1.35, 1.0, 1.0))
    ax.xaxis.pane.set_facecolor((0.96, 0.97, 1.00, 0.42))
    ax.yaxis.pane.set_facecolor((0.95, 1.00, 0.97, 0.34))
    ax.zaxis.pane.set_facecolor((1.00, 0.98, 0.91, 0.28))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor((0.78, 0.80, 0.86, 0.45))
        axis._axinfo['grid']['color'] = (0.62, 0.66, 0.74, 0.28)
        axis._axinfo['grid']['linewidth'] = 0.55
    ax.tick_params(labelsize=8, pad=0)

def _plot_attack_ridge(ax, attack_data, all_k, k_to_color):
    available_k = [k for k in all_k if k in attack_data]
    if not available_k:
        return

    max_len = max(len(attack_data[k]) for k in available_k)
    all_acc = np.concatenate([np.asarray(attack_data[k], dtype=float) for k in available_k])
    z_floor = max(0.0, float(np.nanmin(all_acc)) - 0.08)
    z_top = min(1.0, float(np.nanmax(all_acc)) + 0.08)
    y_positions = np.arange(len(available_k), dtype=float)

    curve_rows = []
    for y_pos, k_val in zip(y_positions, available_k):
        acc = np.asarray(attack_data[k_val], dtype=float)
        xvals = np.arange(len(acc), dtype=float)
        color = k_to_color[k_val]
        curve_rows.append((y_pos, k_val, xvals, acc, color))

    connector_step = max(1, max_len // 5)
    connector_idx = np.r_[np.arange(0, max_len, connector_step), max_len - 1]
    connector_idx = np.unique(connector_idx)
    x_text_pad = max_len * 0.010
    y_text_pad = 0.014
    z_text_pad = max((z_top - z_floor) * 0.014, 0.006)

    for x_idx in connector_idx:
        connector_points = []
        for y_pos, _, _, acc, _ in curve_rows:
            if x_idx < len(acc):
                connector_points.append((y_pos, float(acc[x_idx])))
        if len(connector_points) < 2:
            continue
        ys, zs = zip(*connector_points)
        ax.plot(np.full(len(ys), x_idx), ys, zs,
                color='black', linewidth=0.65, linestyle='--',
                alpha=0.36, zorder=6)

    # Waterfall plots need painter-style ordering in mplot3d: far curtains first,
    # then nearer curtains so the front faces hide back curves.
    for y_pos, _, xvals, acc, color in reversed(curve_rows):
        row_zorder = 10 + (len(curve_rows) - y_pos) * 3
        verts = [[(xvals[0], z_floor), *zip(xvals, acc), (xvals[-1], z_floor)]]
        poly = PolyCollection(
            verts,
            facecolors=[colors.to_rgba(color, 0.28)],
            edgecolors=[colors.to_rgba(color, 0.74)],
            linewidths=0.55,
            zorder=row_zorder,
        )
        ax.add_collection3d(poly, zs=[y_pos], zdir='y')

        ax.plot(xvals, np.full_like(xvals, y_pos), acc,
                color=color, linewidth=1.05, alpha=0.98,
                zorder=row_zorder + 1)

        point_idx = connector_idx[connector_idx < len(acc)]
        if len(point_idx) > 0:
            point_x = xvals[point_idx]
            point_z = acc[point_idx]
            ax.scatter(point_x, np.full(len(point_idx), y_pos), point_z,
                       color=color, edgecolors='none', linewidths=0,
                       marker='o', s=22, depthshade=False,
                       zorder=row_zorder + 1.2)

        for x_idx in point_idx:
            z_val = float(acc[x_idx])
            if x_idx <= connector_idx[0]:
                text_x = x_idx + x_text_pad
                ha = 'left'
            elif x_idx >= connector_idx[-1]:
                text_x = x_idx - x_text_pad
                ha = 'right'
            else:
                text_x = x_idx + x_text_pad * 0.55
                ha = 'left'
            text_y = y_pos + y_text_pad
            text_z = min(z_val + z_text_pad, z_top - z_text_pad * 0.25)
            ax.text(text_x, text_y, text_z,
                    f'{z_val:.2f}', color='#2a1010', fontsize=6.5,
                    ha=ha, va='bottom', zorder=row_zorder + 1.3)

    ax.set_xlim(0, max_len - 1)
    ax.set_ylim(-0.45, len(available_k) - 0.55)
    ax.set_zlim(z_floor, z_top)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'k={k}' for k in available_k])
    ax.set_xlabel('Round', fontsize=9.5, labelpad=2)
    ax.set_ylabel('k', fontsize=9.5, labelpad=2)
    ax.set_zlabel('Accuracy', fontsize=9.5, labelpad=3)
    _style_3d_axis(ax)

def plot_figure(data_iid, iid_label, save_path):
    attacks = [a for a in ATTACK_ORDER if a in data_iid]
    if not attacks:
        print(f"[WARN] No attacks found for {iid_label}")
        return

    # Collect all k values across attacks
    all_k = set()
    for attack in attacks:
        all_k.update(data_iid[attack].keys())
    all_k = sort_k(list(all_k))
    k_to_color = build_k_colors(all_k)
    fig = plt.figure(figsize=(26, 7.2), dpi=300)
    axes = [fig.add_subplot(1, 4, i + 1, projection='3d') for i in range(4)]

    for ax_idx, attack in enumerate(ATTACK_ORDER):
        ax = axes[ax_idx]
        if attack in data_iid:
            _plot_attack_ridge(ax, data_iid[attack], all_k, k_to_color)
            if ax_idx == len(axes) - 1:
                # The rightmost mplot3d z-label is prone to being dropped by
                # bbox_inches='tight'. Use an axes-anchored label there so the
                # exported figure stays compact and the label is retained.
                ax.set_zlabel('')
                ax.text2D(1.035, 0.52, 'Accuracy', transform=ax.transAxes,
                          rotation=90, ha='center', va='center',
                          fontsize=9.5)
            legend_handles = [
                Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor=k_to_color[k_val],
                       markeredgecolor='none', markersize=6.5,
                       label=f'k={k_val}')
                for k_val in all_k
                if k_val in data_iid[attack]
            ]
            ax.legend(handles=legend_handles, loc='upper left',
                      bbox_to_anchor=(0.02, 0.98), bbox_transform=ax.transAxes,
                      frameon=False, fontsize=8.5, handlelength=0.8,
                      handletextpad=0.35, borderaxespad=0.0, labelspacing=0.35)
        else:
            ax.text2D(0.34, 0.5, 'No data', transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
        ax.text2D(0.5, -0.025, ATTACK_DISPLAY.get(attack, attack),
                  transform=ax.transAxes, ha='center', va='top',
                  fontsize=15, weight='bold')


    # Keep the exported image tight for paper use; the rightmost z-label is
    # drawn as axes text above because mplot3d can drop it from tight bboxes.
    plt.subplots_adjust(left=0.02, right=0.95, bottom=0.07, top=0.94, wspace=0.08)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.08, dpi=300)
    print(f"[INFO] Figure saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    data = organize_data(ROOT_DIR)

    for iid_flag in ['iid', 'noniid']:
        if iid_flag not in data:
            continue
        iid_label = 'IID' if iid_flag == 'iid' else 'Non-IID'
        save_path = os.path.join(ROOT_DIR, f'abl_k_{iid_flag}.png')
        plot_figure(data[iid_flag], iid_label, save_path)
