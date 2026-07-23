import os
import re
import torch
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import matplotlib

# --- 配置 ---
root_base = '/home/lhxu/FL/results_mnist_ne_727_bz_50_lr_0.1_500_25'
output_dir = os.path.join(root_base, 'outputs_loss')
os.makedirs(output_dir, exist_ok=True)

# --- 自定义配色与标记 ---
color_map = {
    'geometric': 'blue',
    'huber': '#D92F1E',
    'median': 'green',
    'mean': '#EF9163',
    'krum': '#6F6B84'
}

marker_map = {
    'krum': 'P',
    'geometric': 'X',
    'median': 's',
    'mean': 'D',
    'huber': 'v'
}

# --- 支持的数据集与攻击类型 ---
datasets = ['COVTYPE', 'IJCNN1', 'MNIST']
attacks = ['withoutatt', 'signflipping', 'zerogradient', 'randomnoise']

# --- 收集 loss 数据 ---
all_losses_per_dataset = defaultdict(list)
loss_results = {}
loss_rows = []
raw_loss_rows = []

for folder in os.listdir(root_base):
    folder_path = os.path.join(root_base, folder)
    if not os.path.isdir(folder_path):
        continue

    folder_lower = folder.lower()
    dataset = next((ds for ds in datasets if ds.lower() in folder_lower), None)
    attack = next((atk for atk in attacks if atk in folder_lower), None)
    aggregator = next((agg for agg in color_map if agg in folder_lower), None)

    if not (dataset and attack and aggregator):
        print(f"[跳过] 无法识别：{folder}")
        continue

    print(f"[处理] {folder} -> {dataset}, {attack}, {aggregator}")

    # 找出最大的 round 的 state 文件
    state_files = []
    for f in os.listdir(folder_path):
        match = re.match(r'state_round_(\d+)\.pth', f)
        if match:
            round_num = int(match.group(1))
            state_files.append((round_num, f))
    state_files.sort()

    # 加载最大 round 的 loss（最后一个 round）
    loss_curve = []
    for _, f in reversed(state_files):
        try:
            data = torch.load(os.path.join(folder_path, f), map_location='cpu')
            loss_list = data.get('history', {}).get('loss', [])
            if loss_list:
                loss_curve = loss_list
                break
        except Exception as e:
            print(f"[错误] 读取 {f} 失败: {e}")

    if loss_curve:
        loss_results[(dataset, attack, aggregator)] = loss_curve
        all_losses_per_dataset[dataset].extend(loss_curve)

        raw_row = {
            'dataset': dataset,
            'attack': attack,
            'aggregator': aggregator
        }
        for idx, val in enumerate(loss_curve):
            raw_row[f'step_{idx}'] = val
        raw_loss_rows.append(raw_row)

# --- 计算每个 dataset 的最小 loss ---
f_min = {dataset: min(losses) for dataset, losses in all_losses_per_dataset.items()}
print("自动计算的 f_min：", f_min)

# --- 构建 adjusted loss 表 ---
for (dataset, attack, aggregator), loss_curve in loss_results.items():
    adjusted_loss = [l - f_min[dataset] for l in loss_curve]
    row = {
        'dataset': dataset,
        'attack': attack,
        'aggregator': aggregator
    }
    for idx, val in enumerate(adjusted_loss):
        row[f'step_{idx}'] = val
    loss_rows.append(row)

# --- 保存 CSV ---
loss_df = pd.DataFrame(loss_rows)
loss_df.to_csv(os.path.join(output_dir, 'loss_results.csv'), index=False)
raw_loss_df = pd.DataFrame(raw_loss_rows)
raw_loss_df.to_csv(os.path.join(output_dir, 'raw_loss_all.csv'), index=False)
print(f"[✓] Loss CSV 文件已保存到：{output_dir}")

# --- 配置 matplotlib ---
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

round_steps = 100  # 每轮训练的 step 数

# --- 绘图 ---
for dataset in datasets:
    if dataset not in all_losses_per_dataset:
        continue

    # 图1：adjusted loss
    fig1, axes1 = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 4), squeeze=False)
    for j, attack in enumerate(attacks):
        ax = axes1[0][j]
        for aggregator in color_map:
            key = (dataset, attack, aggregator)
            if key in loss_results:
                # loss = [l - f_min[dataset] for l in loss_results[key]]
                loss = loss_results[key]
                # print(loss)
                x = [(i * round_steps) / 1000 for i in range(len(loss))]
                ax.plot(x, loss,
                        label=aggregator.capitalize(),
                        color=color_map[aggregator],
                        marker=marker_map[aggregator],
                        markevery=max(1, len(x) // 10),
                        linewidth=2)
        ax.set_title(f'{dataset} - {attack}')
        ax.set_xlabel('Iteration (k / 1000)')
        ax.set_ylabel('Loss (log)')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_facecolor('white')
    handles, labels = axes1[0][0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', ncol=len(color_map),
                frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(output_dir, f'adjusted_loss_curve_{dataset.lower()}.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[✓] Adjusted Loss 图已保存到：{path1}")

    # 图2：原始 loss
    fig2, axes2 = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 4), squeeze=False)
    for j, attack in enumerate(attacks):
        ax = axes2[0][j]
        for aggregator in color_map:
            key = (dataset, attack, aggregator)
            if key in loss_results:
                loss = loss_results[key]
                x = [(i * round_steps) / 1000 for i in range(len(loss))]
                ax.plot(x, loss,
                        label=aggregator.capitalize(),
                        color=color_map[aggregator],
                        marker=marker_map[aggregator],
                        markevery=max(1, len(x) // 10),
                        linewidth=2)
        ax.set_title(f'{dataset} - {attack}')
        ax.set_xlabel('Iteration (k / 1000)')
        ax.set_ylabel('Raw Loss (log)')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_facecolor('white')
    handles2, labels2 = axes2[0][0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='upper center', ncol=len(color_map),
                frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path2 = os.path.join(output_dir, f'raw_loss_curve_{dataset.lower()}.png')
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[✓] Raw Loss 图已保存到：{path2}")
