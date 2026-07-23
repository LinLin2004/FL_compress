import os
import csv
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 可配置参数 ===
attack_type = 'random_noise'  # 可选：withoutatt, sign_flipping, random_noise, zeroGrad
dataset_name = 'IJCNN1'   #COVTYPR or IJCNN1
root_dir = f'/home/lhxu/FL/results_{attack_type.lower()}_ab715'

# === 创建输出目录 ===
output_dir = f'/home/lhxu/FL/outputs_heatmap'
os.makedirs(output_dir, exist_ok=True)

# === 初始化 ===
tau1_list, tau2_list, min_losses = [], [], []

# === 遍历子目录 ===
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    match = re.match(r"results_([0-9.]+)_([0-9.]+)", folder)
    if not match:
        continue
    tau1, tau2 = float(match.group(1)), float(match.group(2))

    # 动态匹配子目录（忽略大小写）
    sub_path = None
    for sub in os.listdir(folder_path):
        if dataset_name.lower() in sub.lower() and attack_type.lower() in sub.lower():
            sub_path = os.path.join(folder_path, sub)
            break
    if not sub_path or not os.path.isdir(sub_path):
        print(f"[跳过] 无有效子目录：{folder_path}")
        continue

    # 读取最小 loss
    min_loss = float('inf')
    for file in os.listdir(sub_path):
        if file.startswith('state_round_') and file.endswith('.pth'):
            try:
                data = torch.load(os.path.join(sub_path, file), map_location='cpu')
                loss_list = data.get('history', {}).get('loss', [])
                if loss_list:
                    min_val = min(loss_list)
                    min_loss = min(min_loss, min_val)
            except Exception as e:
                print(f"[错误] 读取失败 {file}: {e}")

    if min_loss < float('inf'):
        tau1_list.append(tau1)
        tau2_list.append(tau2)
        min_losses.append(min_loss)
    else:
        print(f"[跳过] 未提取有效 loss: {folder_path}")

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

# === 保存 CSV 到输出目录 ===
csv_path = os.path.join(output_dir, f"{dataset_name}_min_loss_table_{attack_type}.csv")
with open(csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["tau1", "tau2", "min_loss"])
    for t1, t2, loss in zip(tau1_list, tau2_list, min_losses):
        writer.writerow([t1, t2, round(loss, 6)])
print(f"[✔] CSV 已保存：{csv_path}")

# === 绘图并保存到输出目录 ===
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
plt.title(f'Minimum Loss Heatmap ({dataset_name}_katyusha_huber_{attack_type})')
plt.tight_layout()

png_path = os.path.join(output_dir, f"heatmap_min_loss_{attack_type}.png")
plt.savefig(png_path)
plt.show()
print(f"[✔] 热力图已保存：{png_path}")
