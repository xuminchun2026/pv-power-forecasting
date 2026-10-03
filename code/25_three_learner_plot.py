#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨站迁移：三种 learner 上的收益横向对照（图 19）
==================================================
为什么单独出一张图（2026-10-03）
--------------------------------
树模型过去只有 LightGBM 一种（打印在 code/06 里，报告 §3.7 也明写「四种做法
统一使用 LightGBM」），而 §3.2 的主实验里 XGBoost 是和 LightGBM 一起跑过的。
所以「多站联合训练最优」这条结论此前只有一类树模型在撑。

跑完 code/24_xgb_transfer_loso.py 之后，这里把 LightGBM / XGBoost / LSTM
三种 learner 的 B（预训练+微调）、C（零样本）、D（多站联合）放一张图里，
看第三条证据链成不成立。

输入（三个 CSV 列名一致：A从零_MAE / B微调_MAE / C零样本_MAE / D联合_MAE）
  results/transfer_loso_results.csv      LightGBM
  results/xgb_transfer_results.csv       XGBoost（code/24 产出）
  results/lstm_transfer_results.csv      LSTM
输出
  results/three_learner_gain.csv
  figures/fig19_three_learner.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 中文标题/轴标签必须显式指定字体，否则 matplotlib 出图全是方框
plt_rc = {"font.sans-serif": ["Arial Unicode MS", "Heiti SC", "SimHei"],
          "axes.unicode_minus": False}

BASE = os.path.expanduser("~/Desktop/电气club作业")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(RES, exist_ok=True)

A = "A从零_MAE"
COLS = [("B 预训练+微调", "B微调_MAE"),
        ("C 零样本迁移", "C零样本_MAE"),
        ("D 多站联合", "D联合_MAE")]


def gain(tb, col):
    """提升率 = (A - X) / A，正数为改善"""
    return ((tb[A] - tb[col]) / tb[A] * 100)


def main():
    import matplotlib.pyplot as plt
    plt.rcParams.update(plt_rc)

    src = {"LightGBM(树)": "transfer_loso_results.csv",
           "XGBoost(树)": "xgb_transfer_results.csv",
           "LSTM(神经网络)": "lstm_transfer_results.csv"}
    tabs = {}
    for name, fn in src.items():
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少 {fn}，请先跑对应的迁移脚本")
        tabs[name] = pd.read_csv(p, encoding="utf-8-sig")

    rows = []
    for name, tb in tabs.items():
        for tag, col in COLS:
            v = gain(tb, col)
            rows.append(dict(策略=tag, learner=name,
                             平均提升率=float(v.mean()),
                             有效站数=int((v > 0).sum()), 站数=len(v)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "three_learner_gain.csv"),
             index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    colors = {"LightGBM(树)": "#27ae60", "XGBoost(树)": "#2e86c1",
              "LSTM(神经网络)": "#c0392b"}
    # 先按策略分组，组内按 learner 排开
    schemes = [c[0] for c in COLS]
    learners = list(src.keys())
    xv = np.arange(len(schemes))
    w = 0.27

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for i, lr in enumerate(learners):
        sub = R[R.learner == lr].set_index("策略").loc[schemes]
        v = sub["平均提升率"].values
        pos = xv + (i - 1) * w
        ax.bar(pos, v, w, label=lr, color=colors[lr], alpha=0.92)
        for x, val in zip(pos, v):
            ax.text(x, val, f"{val:+.2f}%", ha="center", fontsize=8.5,
                    va="bottom" if val >= 0 else "top")
    ax.set_xticks(xv)
    ax.set_xticklabels(schemes, fontsize=10)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("相对「从零训练」的误差降幅（%）")
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.legend(fontsize=9, ncol=3, frameon=False)
    ax.set_title("图19  三种 learner 上跨站策略的收益对比（十站平均）", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig19_three_learner.png"), dpi=160)
    plt.close(fig)

    print("图19 三 learner 对比 OK")
    print(R.round(4).to_string(index=False))

    # ---------- 文字小结（写报告时直接用）----------
    print()
    print("-" * 60)
    print("逐策略的三 learner 排序与有效站数")
    print("-" * 60)
    for tag in schemes:
        s = R[R.策略 == tag].sort_values("平均提升率", ascending=False)
        parts = [f"{r.learner} {r.平均提升率:+.2f}%（{r.有效站数}/{r.站数} 站）"
                 for r in s.itertuples()]
        print(f"  {tag:<12s} " + "  |  ".join(parts))
    print()
    for tag, col in COLS:
        pass
    # 树模型两类的差距（同策略下 LightGBM vs XGBoost）
    for tag, col in COLS:
        lg = gain(tabs["LightGBM(树)"], col).mean()
        xg = gain(tabs["XGBoost(树)"], col).mean()
        print(f"  {tag:<12s} LightGBM {lg:+.2f}%  vs  XGBoost {xg:+.2f}%   "
              f"差 {xg - lg:+.2f} 个百分点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
