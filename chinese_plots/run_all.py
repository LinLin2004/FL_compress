import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

scripts = [
    'plot_comparison_acc_zh.py',
    'plot_comparison_loss_zh.py',
    'plot_ablation_acc_zh.py',
    'plot_ablation_loss_zh.py',
]

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"[ERROR] {script} exited with code {result.returncode}")

print(f"\n{'='*60}")
print("所有脚本执行完毕，图像输出到:")
print(f"  {SCRIPT_DIR}")
print(f"{'='*60}")
