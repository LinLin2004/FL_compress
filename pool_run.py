#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import os.path as osp
import threading
import datetime

# --- 在这里修改你要运行的命令列表 ---
# 这是一个示例列表，包含了一些会耗费不同时间且可能成功或失败的命令
COMMANDS_TO_RUN = []
cfg_dir = 'configs/cfg_main_exp'
all_configs = sorted([osp.join(cfg_dir, cfg_name) for cfg_name in os.listdir(cfg_dir) if cfg_name.endswith('.yaml')])
# all_configs = []
# all_configs += [
# ]]
for config in all_configs:
    cmd = f'python run_experiment.py --config {config}'
    COMMANDS_TO_RUN.append(cmd)


# --- 在这里修改线程池的大小 ---
# 线程池的最大工作线程数。可以根据你的 CPU 和任务类型调整。
# 对于 I/O 密集型任务，可以设置得比 CPU 核心数多。
MAX_WORKERS = 4


class ProgressTracker:
    """线程安全的进度追踪器"""

    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.running = 0
        self.lock = threading.Lock()

    def on_start(self, command_id, command):
        with self.lock:
            self.running += 1
            self._print_progress(command_id, command, "开始")

    def on_finish(self, command_id, status):
        with self.lock:
            self.running -= 1
            self.completed += 1
            self._print_progress(command_id, None, "完成", status)

    def _print_progress(self, command_id, command, event, status=None):
        remaining = self.total - self.completed - self.running
        header = f"[{self.completed}/{self.total}] 正在跑={self.running} | 剩余={remaining}"
        if event == "开始":
            # 只取配置文件名，缩短显示
            short_cmd = os.path.basename(command.split('--config ')[-1]) if '--config ' in command else command[:50]
            print(f"{header} | ▶ 任务{command_id} 开始: {short_cmd}")
        else:
            status_icon = "✓" if status == "成功" else "✗"
            print(f"{header} | {status_icon} 任务{command_id} {status}")


def run_command(command: str, command_id: int, tracker: ProgressTracker):
    """
    执行单个 shell 命令并返回结果。

    参数:
    - command (str): 要执行的命令字符串。
    - command_id (int): 命令的唯一标识符，用于跟踪。
    - tracker (ProgressTracker): 进度追踪器。

    返回:
    - dict: 包含命令执行结果的字典。
    """
    tracker.on_start(command_id, command)

    start_time = time.time()

    # 为每个子进程创建日志文件，保存 stdout 和 stderr
    log_dir = osp.join(osp.dirname(osp.abspath(__file__)), "results", "pool_logs")
    os.makedirs(log_dir, exist_ok=True)
    # 从命令中提取配置文件名作为日志文件名
    if '--config ' in command:
        cfg_name = os.path.basename(command.split('--config ')[-1]).replace('.yaml', '')
    else:
        cfg_name = f"task_{command_id}"
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_path = osp.join(log_dir, f"{cfg_name}_{timestamp}.log")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )

        end_time = time.time()
        status = "成功" if result.returncode == 0 else "失败"

        # 将 stdout 和 stderr 写入日志文件
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"命令: {command}\n")
            f.write(f"返回码: {result.returncode}\n")
            f.write(f"耗时: {end_time - start_time:.1f}s\n")
            f.write(f"{'='*60}\n")
            f.write(f"=== STDOUT ===\n")
            f.write(result.stdout)
            f.write(f"\n=== STDERR ===\n")
            f.write(result.stderr)

        return {
            "id": command_id,
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "duration": end_time - start_time,
            "status": status,
            "log_path": log_path,
        }
    except Exception as e:
        end_time = time.time()
        # 异常也写日志
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"命令: {command}\n")
            f.write(f"Python 异常: {e}\n")
            f.write(f"耗时: {end_time - start_time:.1f}s\n")
        return {
            "id": command_id,
            "command": command,
            "return_code": -1,
            "stdout": "",
            "stderr": f"执行时发生 Python 异常: {e}",
            "duration": end_time - start_time,
            "status": "异常",
            "log_path": log_path,
        }

def main():
    """
    主函数，创建线程池并执行所有命令。
    """
    total = len(COMMANDS_TO_RUN)
    print(f"准备执行 {total} 个命令，使用 {MAX_WORKERS} 个工作线程。")
    print("-" * 60)
    
    tracker = ProgressTracker(total)
    start_total_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_command = {
            executor.submit(run_command, cmd, i, tracker): (i, cmd) 
            for i, cmd in enumerate(COMMANDS_TO_RUN, 1)
        }
        
        for future in as_completed(future_to_command):
            command_id, command_str = future_to_command[future]
            try:
                result_data = future.result()
                results.append(result_data)
                tracker.on_finish(command_id, result_data['status'])
            except Exception as exc:
                results.append({
                    "id": command_id,
                    "command": command_str,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": str(exc),
                    "duration": 0,
                    "status": "异常"
                })
                tracker.on_finish(command_id, "异常")

    end_total_time = time.time()
    
    print("-" * 60)
    
    # 统计
    success_count = sum(1 for r in results if r['status'] == '成功')
    fail_count = sum(1 for r in results if r['status'] == '失败')
    error_count = sum(1 for r in results if r['status'] == '异常')
    print(f"全部完成! 成功={success_count} 失败={fail_count} 异常={error_count} 总计={total}")
    print()
    
    # 对结果按 ID 排序
    results.sort(key=lambda x: x['id'])
    
    # 只打印非成功的任务详情
    failed_results = [r for r in results if r['status'] != '成功']
    if failed_results:
        print("以下任务未成功：")
        for res in failed_results:
            short_cmd = os.path.basename(res['command'].split('--config ')[-1]) if '--config ' in res['command'] else res['command'][:50]
            print(
                f"  任务{res['id']} [{res['status']}] (返回码: {res['return_code']}) {short_cmd} "
                f"耗时: {res['duration']:.1f}s"
            )
            # 打印 stderr 的最后几行，方便快速定位问题
            if res['stderr']:
                stderr_lines = res['stderr'].splitlines()
                # 最多显示最后 5 行
                tail = stderr_lines[-5:] if len(stderr_lines) > 5 else stderr_lines
                print(f"    stderr (最后{len(tail)}行):")
                for line in tail:
                    print(f"      {line}")
            # 打印日志文件路径，方便查看完整输出
            log_path = res.get('log_path', '')
            if log_path and os.path.exists(log_path):
                print(f"    完整日志: {log_path}")
        print()

    print(f"总耗时: {end_total_time - start_total_time:.1f} 秒")

if __name__ == "__main__":
    main()
