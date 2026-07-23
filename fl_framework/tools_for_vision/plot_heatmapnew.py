import os
import csv
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 可配置参数 ===
target_dataset = 'covtype'  # 数据集名：covtype 或 IJCNN1
root_dir = 'results_covtype_tau'
output_base = f'/home/lhxu/FL/outputs_heatmap'

# === 获取所有攻击类型 ===
all_folders = os.listdir(root_dir)
attack_types = set()

for folder in all_folders:
    folder_lower = folder.lower()
    if target_dataset in folder_lower:
        match = re.match(rf"{target_dataset}_katyusha_huber_([a-zA-Z0-9]+)_\d", folder_lower)
        if match:
            attack_types.add(match.group(1))

attack_types = sorted(attack_types)
print(f"[✔] 发现攻击类型：{attack_types}")

# === 遍历每种攻击类型 ===
for attack_type in attack_types:
    print(f"\n[处理中] 数据集: {target_dataset} | 攻击类型: {attack_type}")
    output_dir = os.path.join(output_base, f"{target_dataset}_{attack_type}")
    os.makedirs(output_dir, exist_ok=True)

    tau1_list, tau2_list, min_losses = [], [], []

    for folder in all_folders:
        folder_lower = folder.lower()
        if f"{target_dataset}_katyusha_huber_{attack_type}" not in folder_lower:
            continue

        match = re.search(r'_(\d\.\d+)_([\d.]+)$', folder)
        if not match:
            print(f"[跳过] 未能解析 tau1, tau2：{folder}")
            continue
        tau1, tau2 = float(match.group(1)), float(match.group(2))
        folder_path = os.path.join(root_dir, folder)

        min_loss = float('inf')
        for file in os.listdir(folder_path):
            if file.startswith('state_round_') and file.endswith('.pth'):
                try:
                    data = torch.load(os.path.join(folder_path, file), map_location='cpu')
                    loss_list = data.get('history', {}).get('loss', [])
                    if loss_list:
                        min_val = min(loss_list)
                        min_loss = min(min_loss, min_val)
                except Exception as e:
                    print(f"[错误] 读取 {file}: {e}")

        if min_loss < float('inf'):
            tau1_list.append(tau1)
            tau2_list.append(tau2)
            min_losses.append(min_loss)
        else:
            print(f"[跳过] 无有效 loss: {folder_path}")

    if not min_losses:
        print(f"[跳过] 无有效结果，跳过绘图")
        continue

    # === 构造热力图矩阵 ===
    unique_tau1 = sorted(set(tau1_list))
    unique_tau2 = sorted(set(tau2_list))
    heatmap_matrix = np.full((len(unique_tau2), len(unique_tau1)), np.nan)

    tau1_idx = {v: i for i, v in enumerate(unique_tau1)}
    tau2_idx = {v: i for i, v in enumerate(unique_tau2)}

    for t1, t2, loss in zip(tau1_list, tau2_list, min_losses):
        i = tau2_idx.get(t2)
        j = tau1_idx.get(t1)
        if i is not None and j is not None:
            heatmap_matrix[i, j] = loss

    # === 保存 CSV ===
    csv_path = os.path.join(output_dir, f"{target_dataset}_min_loss_table_{attack_type}.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["tau1", "tau2", "min_loss"])
        for t1, t2, loss in zip(tau1_list, tau2_list, min_losses):
            writer.writerow([t1, t2, round(loss, 6)])
    print(f"[✔] CSV: {csv_path}")

    # === 绘图并保存 ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=unique_tau1,
        yticklabels=unique_tau2,
        cmap="YlOrRd_r"
    )
    plt.xlabel('tau1')
    plt.ylabel('tau2')
    plt.title(f'Minimum Loss Heatmap ({target_dataset}_katyusha_huber_{attack_type})')
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{target_dataset}_heatmap_min_loss_{attack_type}.png")
    plt.savefig(png_path)
    plt.close()
    print(f"[✔] 热力图: {png_path}")
