import os
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
import re

ROOT_DIR = '/data-store/xulihan/FL_compress/results/new_abl_r'

# Parse r value from dirname like cifar10_adamk_compress4_clip_k469_r1_withoutatt_iid
def parse_dirname(dirname):
    m = re.match(r'cifar10_adamk_compress4_clip_k\d+_r(\d+)_(\w+)_(iid|noniid)', dirname)
    if not m:
        return None
    r_val = m.group(1)
    attack = m.group(2)
    iid_flag = m.group(3)
    return r_val, attack, iid_flag

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
    """Returns: data[iid_flag][attack][r_val] = acc_list"""
    data = {}
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for subdir in subdirs:
        parsed = parse_dirname(subdir)
        if parsed is None:
            print(f"[WARN] Skipping '{subdir}'")
            continue
        r_val, attack, iid_flag = parsed
        folder_path = os.path.join(root_dir, subdir)
        acc_list = read_acc_list(folder_path)
        if not acc_list:
            print(f"[WARN] No data in '{subdir}'")
            continue
        data.setdefault(iid_flag, {})
        data[iid_flag].setdefault(attack, {})
        data[iid_flag][attack][r_val] = acc_list
        print(f"[INFO] Loaded: iid={iid_flag}, attack={attack}, r={r_val}, epochs={len(acc_list)}")
    return data

ATTACK_DISPLAY = {
    'withoutatt': 'No Attack',
    'foe': 'FoE',
    'labelflipping': 'Label Flipping',
    'signflipping': 'Sign Flipping',
}

ATTACK_ORDER = ['withoutatt', 'foe', 'labelflipping', 'signflipping']

# Sort r values numerically
def sort_r(r_list):
    return sorted(r_list, key=lambda x: int(x))

REFERENCE_PALETTE = [
    '#0d0887',
    '#6a00a8',
    '#b12a90',
    '#cf3f73',
    '#f06b44',
    '#f2b447',
    '#f9f74a',
]

MARKERS = ['o', 's', '^', 'D', 'v', 'P']

def build_value_colors(values):
    if len(values) <= 1:
        color_indices = [0]
    else:
        color_indices = np.linspace(0, len(REFERENCE_PALETTE) - 1, len(values))
        color_indices = np.rint(color_indices).astype(int)
    return {
        val: REFERENCE_PALETTE[color_indices[i]]
        for i, val in enumerate(values)
    }

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

def _plot_attack_ridge(ax, attack_data, all_values, value_to_color, value_prefix, y_label):
    available_values = [val for val in all_values if val in attack_data]
    if not available_values:
        return

    max_len = max(len(attack_data[val]) for val in available_values)
    all_acc = np.concatenate([np.asarray(attack_data[val], dtype=float) for val in available_values])
    z_floor = max(0.0, float(np.nanmin(all_acc)) - 0.08)
    z_top = min(1.0, float(np.nanmax(all_acc)) + 0.08)
    y_positions = np.arange(len(available_values), dtype=float)

    curve_rows = []
    for y_pos, value in zip(y_positions, available_values):
        acc = np.asarray(attack_data[value], dtype=float)
        xvals = np.arange(len(acc), dtype=float)
        color = value_to_color[value]
        curve_rows.append((y_pos, value, xvals, acc, color))

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
    ax.set_ylim(-0.45, len(available_values) - 0.55)
    ax.set_zlim(z_floor, z_top)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'{value_prefix}={val}' for val in available_values])
    ax.set_xlabel('Round', fontsize=9.5, labelpad=2)
    ax.set_ylabel(y_label, fontsize=9.5, labelpad=2)
    ax.set_zlabel('Accuracy', fontsize=9.5, labelpad=3)
    _style_3d_axis(ax)

def plot_figure(data_iid, iid_label, save_path):
    attacks = [a for a in ATTACK_ORDER if a in data_iid]
    if not attacks:
        print(f"[WARN] No attacks found for {iid_label}")
        return

    all_r = set()
    for attack in attacks:
        all_r.update(data_iid[attack].keys())
    all_r = sort_r(list(all_r))

    value_to_color = build_value_colors(all_r)
    fig = plt.figure(figsize=(26, 7.2), dpi=300)
    axes = [fig.add_subplot(1, 4, i + 1, projection='3d') for i in range(4)]

    for ax_idx, attack in enumerate(ATTACK_ORDER):
        ax = axes[ax_idx]
        if attack in data_iid:
            _plot_attack_ridge(ax, data_iid[attack], all_r, value_to_color, 'r', 'r')
            if ax_idx == len(axes) - 1:
                ax.set_zlabel('')
                ax.text2D(1.035, 0.52, 'Accuracy', transform=ax.transAxes,
                          rotation=90, ha='center', va='center',
                          fontsize=9.5)
            legend_handles = [
                Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor=value_to_color[r_val],
                       markeredgecolor='none', markersize=6.5,
                       label=f'r={r_val}')
                for r_val in all_r
                if r_val in data_iid[attack]
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
        save_path = os.path.join(ROOT_DIR, f'abl_r_{iid_flag}.png')
        plot_figure(data[iid_flag], iid_label, save_path)
