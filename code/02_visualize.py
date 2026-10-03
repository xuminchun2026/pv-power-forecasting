# -*- coding: utf-8 -*-
# 2026-10-03：本报告最终改用 PVOD 数据集，本脚本为早期用另一份候选数据集做的探索，已弃用；
# 正式结果见 10_pvod_load_check.py 起的各脚本。
"""
第 3 步：画图（报告用 4 张图）
=========================================================
图1 典型日出力曲线        ：功率 与 辐射 随一天时间的变化（双轴）
图2 全年逐日最大功率曲线  ：看一整年的季节变化
图3 辐射-功率散点图      ：看两者的关系（是否近似直线）
图4 特征-功率 相关系数热力图：一眼看出哪些因子最相关

运行后图片会保存到 ../figures/ 目录，报告里直接插这些 png。
=========================================================
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 中文字体：macOS 用 Arial Unicode MS，否则中文会显示成方块
# ---------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False     # 正常显示负号

# ---------------------------------------------------------
# 0) 路径
# ---------------------------------------------------------
DATA_PATH = "/Users/xuminchun/Desktop/电气club作业/data/ningxia_2019_15min.xlsx"
FIG_DIR = "/Users/xuminchun/Desktop/电气club作业/figures"

os.makedirs(FIG_DIR, exist_ok=True)            # 目录不存在就自动建

TIME_COL = "采样结束时刻"
RAD_COL = "GTI（固定倾角辐射）"
TARGET = "实际功率"

# ---------------------------------------------------------
# 1) 读入 + 简单清洗（把第2步发现的 11 条异常剔掉）
#    GTI=0 却功率>0 的 8 条，和有辐射却功率=0 的 3 条
# ---------------------------------------------------------
df = pd.read_excel(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])

valid = df[((df[RAD_COL] == 0) & (df[TARGET] == 0))
           | ((df[RAD_COL] > 0) & (df[TARGET] > 0))].copy()
print("清洗前:", len(df), " 清洗后:", len(valid),
      " 剔除:", len(df) - len(valid), "条")

# =========================================================
# 图1：典型日出力曲线（双轴）
# =========================================================
# 自动挑一个"发电最充分"的日子 = 阳光最好的晴天，避免硬编码日期
daily_hit = valid.groupby(valid[TIME_COL].dt.date).apply(
    lambda g: (g[TARGET] > 1).sum())
best_day = daily_hit.idxmax()
sub = valid[valid[TIME_COL].dt.date == best_day].sort_values(TIME_COL)

fig, ax1 = plt.subplots(figsize=(10, 4.2))
c1, c2 = "#d62728", "#1f77b4"                  # 红=功率，蓝=辐射
ax1.plot(sub[TIME_COL], sub[TARGET], color=c1, lw=1.8, label="实际功率 (kW)")
ax1.set_xlabel("时间")
ax1.set_ylabel("实际功率 (kW)", color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()                              # 双轴：右边再来一根轴
ax2.plot(sub[TIME_COL], sub[RAD_COL], color=c2, lw=1.5, alpha=0.75,
         label="GTI 固定倾角辐射 (W/m²)")
ax2.set_ylabel("辐射 (W/m²)", color=c2)
ax2.tick_params(axis="y", labelcolor=c2)

plt.title(f"典型晴天出力曲线（{best_day}）")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_typical_day.png"), dpi=150)
plt.close(fig)

# =========================================================
# 图2：全年逐日最大功率曲线
# =========================================================
daily_max = valid.groupby(valid[TIME_COL].dt.date)[TARGET].max()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(daily_max.index, daily_max.values, color="#2ca02c", lw=1.2)
ax.set_xlabel("日期（2019年）")
ax.set_ylabel("当日最大功率 (kW)")
ax.set_title("全年逐日最大出力曲线（季节变化）")
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_daily_max.png"), dpi=150)
plt.close(fig)

# =========================================================
# 图3：辐射-功率 散点图
# =========================================================
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(valid[RAD_COL], valid[TARGET], s=4, alpha=0.25,
           color="#9467bd", edgecolors="none")
ax.set_xlabel("GTI 固定倾角辐射 (W/m²)")
ax.set_ylabel("实际功率 (kW)")
ax.set_title("辐射 vs 功率 散点图")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_scatter.png"), dpi=150)
plt.close(fig)

# =========================================================
# 图4：特征-功率 相关系数热力图
# =========================================================
num_cols = [c for c in valid.columns
            if pd.api.types.is_numeric_dtype(valid[c])
            and c != "月"]
corr = valid[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        val = corr.iloc[i, j]
        if abs(val) > 0.5:      # 只标显著的相关系数，太密会看不清
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("各数值特征与功率的相关系数热力图")
fig.colorbar(im, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_corr.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------
# 顺手把相关系数最大的几个打印出来，写报告用得到
# ---------------------------------------------------------
print("\n与『实际功率』相关性最高的前 8 个特征：")
print(corr[TARGET].drop(TARGET).sort_values(ascending=False).head(8).round(3).to_string())
print("\n图已保存到:", FIG_DIR)
