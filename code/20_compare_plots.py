# -*- coding: utf-8 -*-
"""
任务清单核心第 6 项的可视化：把第 2、3、4、5 项的结果放到同一张图里比。

  fig15  单站内横向对比：朴素基线 / 线性回归 / RF / XGBoost / LightGBM / LSTM
         四个信息设定各一组，回答「复杂模型是否一定优于简单模型」。
  fig16  跨站策略对比：树模型与 LSTM 两种 learner 上的
         从零 / 预训练+微调 / 零样本 / 多站联合，回答「怎样利用多站数据更好」。
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/电气club作业")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")

plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

SETTINGS = ["甲-同时刻实测(lmd)", "甲-同时刻预报(nwp)", "乙-纯历史时序", "丙-同时刻气象+历史"]
SHORT = {"甲-同时刻实测(lmd)": "甲 实测气象",
         "甲-同时刻预报(nwp)": "甲′ 预报气象",
         "乙-纯历史时序": "乙 纯历史",
         "丙-同时刻气象+历史": "丙 气象+历史"}
ORDER = ["朴素基线", "线性回归", "随机森林RF", "XGBoost", "LightGBM", "LSTM"]
COLORS = {"朴素基线": "#95a5a6", "线性回归": "#8e44ad", "随机森林RF": "#2980b9",
          "XGBoost": "#16a085", "LightGBM": "#27ae60", "LSTM": "#c0392b"}


def norm_method(m):
    if str(m).startswith("朴素基线"):
        return "朴素基线"
    return m


def fig15_single_station():
    t = pd.read_csv(os.path.join(RES, "pvod_main_per_station.csv"))
    l = pd.read_csv(os.path.join(RES, "pvod_lstm_single.csv"))
    t["方法"] = t["方法"].map(norm_method)
    df = pd.concat([t[["station", "设定", "方法", "MAE"]],
                    l[["station", "设定", "方法", "MAE"]]], ignore_index=True)
    # 每个设定内，朴素基线可能有两条（固定系数 / Persistence），取较好的那条
    df = (df.groupby(["设定", "方法", "station"], as_index=False)["MAE"].min())
    piv = df.pivot_table(index="设定", columns="方法", values="MAE", aggfunc="mean")
    piv = piv.reindex(SETTINGS)
    piv = piv[[c for c in ORDER if c in piv.columns]]

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    xv = np.arange(len(piv))
    n = len(piv.columns)
    w = 0.82 / n
    for i, m in enumerate(piv.columns):
        ax.bar(xv + (i - (n - 1) / 2) * w, piv[m].values, w,
               label=m, color=COLORS.get(m, "#34495e"), alpha=0.9)
        for x, v in zip(xv + (i - (n - 1) / 2) * w, piv[m].values):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6, rotation=90)
    ax.set_xticks(xv)
    ax.set_xticklabels([SHORT[s] for s in piv.index], fontsize=10)
    ax.set_ylabel("十站平均 MAE（%装机容量）")
    ax.set_ylim(0, max(piv.max().max() * 1.28, 1))
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.legend(fontsize=9, ncol=6, frameon=False)
    ax.set_title("图15  单站内六类方法的十站平均误差（越低越好）", fontsize=12)
    fig.savefig(os.path.join(FIG, "fig15_single_station_compare.png"))
    plt.close(fig)
    print("fig15 单站横向对比 OK")
    print(piv.round(4).to_string())
    return piv


def fig16_transfer():
    tree = pd.read_csv(os.path.join(RES, "transfer_loso_results.csv"))
    lstm = pd.read_csv(os.path.join(RES, "lstm_transfer_results.csv"))

    def gain(tb, col_a, col_x):
        """提升率 = (A - X) / A，正数为改善"""
        return ((tb[col_a] - tb[col_x]) / tb[col_a] * 100)

    # 两个 CSV 的列名一致，都是「A从零_MAE / B微调_MAE / C零样本_MAE / D联合_MAE」
    A, B, C, D = "A从零_MAE", "B微调_MAE", "C零样本_MAE", "D联合_MAE"
    for nm, tb in [("树模型", tree), ("LSTM", lstm)]:
        miss = [c for c in (A, B, C, D) if c not in tb.columns]
        if miss:
            raise KeyError(f"{nm} 结果表缺少列 {miss}，实际列为 {list(tb.columns)}")
    schemes = [("B 预训练+微调", B), ("C 零样本迁移", C), ("D 多站联合", D)]

    rows = []
    for name, col in schemes:
        rows.append(dict(策略=name, learner="LightGBM(树)",
                         提升率=float(gain(tree, A, col).mean())))
    for name, col in schemes:
        rows.append(dict(策略=name, learner="LSTM(神经网络)",
                         提升率=float(gain(lstm, A, col).mean())))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "final_transfer_compare.csv"),
             index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    xv = np.arange(len(schemes))
    w = 0.36
    for i, lr in enumerate(["LightGBM(树)", "LSTM(神经网络)"]):
        sub = R[R.learner == lr].set_index("策略").loc[[s[0] for s in schemes]]
        v = sub["提升率"].values
        ax.bar(xv + (i - 0.5) * w, v, w, label=lr,
               color=["#27ae60", "#c0392b"][i], alpha=0.9)
        for x, val in zip(xv + (i - 0.5) * w, v):
            ax.text(x, val, f"{val:+.2f}%", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=9)
    ax.set_xticks(xv)
    ax.set_xticklabels([s[0] for s in schemes], fontsize=10)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("相对「从零训练」的误差降幅（%）")
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.legend(fontsize=9)
    ax.set_title("图16  两种 learner 上跨站策略的收益对比（十站平均）", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig16_transfer_compare.png"))
    plt.close(fig)
    print("\nfig16 跨站策略对比 OK")
    print(R.round(4).to_string(index=False))
    return R


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig15_single_station()
    print()
    fig16_transfer()
