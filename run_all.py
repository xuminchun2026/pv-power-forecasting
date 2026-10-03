# -*- coding: utf-8 -*-
"""
一键跑通全部实验（第 2 步 ~ 第 5 步）
运行：右键本文件 -> 运行 'run_all'（或终端 python run_all.py）

作用：按顺序调用 code/ 下 6 个分析脚本，中间不用来回切窗口。
      每个脚本仍可单独运行，便于只复现某一个结论。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"

SCRIPTS = [
    ("第2步 数据检查", "01_load_and_check.py"),
    ("第3步 可视化", "02_visualize.py"),
    ("第3b步 目标列性质检验", "02b_power_vs_irradiance.py"),
    ("第4步 建模与评估", "03_model.py"),
    ("第5步 可解释性", "04_interpretability.py"),
    ("第5b步 温度/云量消融", "05_feature_ablation.py"),
]

if __name__ == "__main__":
    print("=" * 60)
    print("一键复现：按时间顺序依次运行全部分析脚本")
    print("=" * 60)

    for name, fname in SCRIPTS:
        path = CODE_DIR / fname
        if not path.exists():
            print(f"[跳过] {name}：找不到 {fname}")
            continue
        print(f"\n{'=' * 60}\n>>> 正在运行：{name}  ({fname})\n{'=' * 60}")
        r = subprocess.run([sys.executable, str(path)])
        if r.returncode != 0:
            print(f"\n[中断] {fname} 退出码 {r.returncode}，后续脚本已停止。")
            break

    print(f"\n{'=' * 60}")
    print("[OK] 全部跑完。结果在 results/，图在 figures/。")
    print("=" * 60)
