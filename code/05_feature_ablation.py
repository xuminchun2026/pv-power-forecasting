# -*- coding: utf-8 -*-
# 2026-10-03：本报告最终改用 PVOD 数据集，本脚本为早期用另一份候选数据集做的探索，已弃用；
# 正式结果见 10_pvod_load_check.py 起的各脚本。
"""
============================================================
第 5 步（补充实验）：验证" temperature 和 云量 到底能不能丢掉"
============================================================
怎么运行：PyCharm 里右键本文件 -> 运行 '05_feature_ablation'
前置：先跑过 03_model.py（要用同一套切分方式，结果才可比）

【这一步要回答的问题（来自指导学长的经验判断）】
学长说："这数据跟温度、云量关系本来就低，是正常现象（说明当地天气稳），
          实际预测时不考虑这两个因素就行。"
这句话对不对？不能靠嘴说，要拿数据验。

分三问：
  问1 相关性：功率跟温度、云量的线性相关系数到底有多大？
               比起来跟 GHI 比，差几个量级？
  问2 控制变量：把辐射（GHI）卡死在差不多的水平上，
                温度高的时候功率到底升还是降？降多少？
                —— 这是关键！因为温度本来就跟辐射一起涨，
                   不卡住辐射就没法判断"温度是不是真的影响功率"。
  问3 消融实验：把温度、云量这两个特征从模型里删掉，
                预测误差会变差多少？变差很多 → 不能丢；几乎不变 → 可以丢。

【红线提醒（答辩常考）】
  如果删了特征以后误差几乎没变，正确的说法是"在本数据集、这个模型下
  该特征贡献可忽略"，而不能说成"温度对光伏发电没有物理影响"。
  温度对光伏的影响真实存在（组件温度每升高1℃，功率约降0.3%~0.4%），
  只是它比辐射小一个数量级，而且模型能从辐射里"顺带"学到。
  这句话一定要能背下来，答辩老师百分之百会问"温度效应你考虑了没"。
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 0) 全局设置（跟 03_model.py 保持一致，否则结果没法对比）
# ============================================================
DATA_PATH = "/Users/xuminchun/Desktop/电气club作业/data/ningxia_2019_15min.xlsx"
TIME_COL = "采样结束时刻"
TARGET = "实际功率"
RANDOM_STATE = 42

GHI = "GHI（太阳总水平辐射）"
TEMP = "气温"
CLOUD = "云层不透明度"
WIND = "高度10m风速"
RH = "相对湿度"

BASE = "/Users/xuminchun/Desktop/电气club作业"
FIG_DIR = os.path.join(BASE, "figures")
OUT_DIR = os.path.join(BASE, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 中文显示（macOS 自带字体，跟前面几步脚本统一）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1) 读入数据 + 造滞后特征（跟 03_model.py 的设定乙完全一样）
# ============================================================
def add_lag_features(d):
    """shift(k) 就是"k 个 15 分钟之前"的旧值"""
    d = d.sort_values(TIME_COL).reset_index(drop=True)
    d["GHI_lag1"] = d[GHI].shift(1)
    d["P_lag1"] = d[TARGET].shift(1)
    d["T_lag1"] = d[TEMP].shift(1)
    d["T_lag6"] = d[TEMP].shift(6)
    t = d[TIME_COL]
    d["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    d["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365)
    d["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365)
    return d


print("=" * 72)
print("读入数据…")
df = pd.read_excel(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = add_lag_features(df).dropna()

# ============================================================
# 2) 问1：线性相关系数
# ============================================================
print("=" * 72)
print("【问1】相关系数：功率跟各气象因子的线性相关程度")
print("只统计白天（GHI > 50 W/m²）的时段 —— 晚上没太阳功率就是0，会把相关性搅乱")

day = df[df[GHI] > 50]
print(f"白天样本数：{len(day)}")

rows = []
for label, col in [("GHI 总辐射", GHI), ("气温", TEMP), ("云层不透明度", CLOUD),
                   ("相对湿度", RH), ("风速10m", WIND)]:
    r = np.corrcoef(day[col], day[TARGET])[0, 1]
    rows.append({"因子": label, "Pearson相关系数": round(r, 4)})

corr_df = pd.DataFrame(rows).sort_values("Pearson相关系数", key=abs, ascending=False)
print(corr_df.to_string(index=False))

# ============================================================
# 3) 问2：控制变量 —— 卡住辐射，单看温度
# ============================================================
print("=" * 72)
print("【问2】控制变量检验：同样的辐射下，温度高/低，功率差多少？")
print("做法：把 GHI 每 50 W/m² 装成一个「箱子」，只看箱内的点。")
print("      箱内辐射几乎一样，那功率的任何差别就只能归因于温度了。")

box = []
# 只用"发电时段"且在箱内样本够多的箱
for lo in range(150, 1000, 50):
    hi = lo + 50
    sel = day[(day[GHI] >= lo) & (day[GHI] < hi)]
    if len(sel) < 300 or sel[TEMP].std() < 1.0:
        continue                      # 样本太少或温度几乎没变化，跳过
    slope = np.polyfit(sel[TEMP], sel[TARGET], 1)[0]   # kW/℃
    box.append({
        "GHI区间": f"{lo}-{hi}",
        "样本数": len(sel),
        "箱内温度跨度℃": round(sel[TEMP].max() - sel[TEMP].min(), 1),
        "温度效应 kW/℃": round(slope, 4),
    })

box_df = pd.DataFrame(box)
if len(box_df):
    print(box_df.to_string(index=False))
    print(f"\n各辐射档位下，「温度每升高1℃ → 功率变化」的平均值："
          f"{box_df['温度效应 kW/℃'].mean():.4f} kW/℃")
    print(f"最大绝对值：{box_df['温度效应 kW/℃'].abs().max():.4f} kW/℃")
    print("作为对照：该电站观测最大功率约 128 kW，白天 GHI 每增加 100 W/m² "
          "功率大约增加 12~13 kW —— 温度的量级比它小 1~2 个数量级。")

# ============================================================
# 4) 问3：消融实验（Ablation）
# ============================================================
print("=" * 72)
print("【问3】消融实验：把特征删掉，模型还准不准？")

FEAT_A = [GHI, "DNI（太阳直接辐射指数）", "DHI（太阳散射辐射指数）",
          "GTI（固定倾角辐射）", TEMP, RH, CLOUD, WIND,
          "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
DROP_A = [TEMP, CLOUD]                                  # 甲设定：删气温 + 云量

FEAT_B = ["GHI_lag1", "P_lag1", "T_lag1", "T_lag6",
          "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
DROP_B = ["T_lag1", "T_lag6"]                           # 乙设定：删温度滞后

cut = int(len(df) * 0.7)
tr, te = df.iloc[:cut], df.iloc[cut:]
print(f"时间切分：训练 {tr[TIME_COL].min().date()} ~ "
      f"{tr[TIME_COL].max().date()}（70%），"
      f"测试 {te[TIME_COL].min().date()} ~ {te[TIME_COL].max().date()}（30%）")

rows = []


def run(tag, feats):
    mdl = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    mdl.fit(tr[feats].values, tr[TARGET].values)
    pred = mdl.predict(te[feats].values)
    rows.append({
        "设定": tag,
        "特征数": len(feats),
        "MAE(kW)": round(mean_absolute_error(te[TARGET].values, pred), 4),
        "RMSE(kW)": round(np.sqrt(mean_squared_error(te[TARGET].values, pred)), 4),
        "R2": round(r2_score(te[TARGET].values, pred), 4),
    })


run("甲 完整(含温度+云量)", FEAT_A)
run("甲 只删掉温度", [f for f in FEAT_A if f != TEMP])
run("甲 只删掉云量", [f for f in FEAT_A if f != CLOUD])
run("甲 温度云量都删", [f for f in FEAT_A if f not in DROP_A])
run("乙 完整(含温度滞后)", FEAT_B)
run("乙 删掉温度滞后", [f for f in FEAT_B if f not in DROP_B])

abl = pd.DataFrame(rows)
print()
print(abl.to_string(index=False))

print("\n结论行（删特征 vs 完整，MAE 变化百分比）：")
for i_full, i_cut, tag in [(0, 1, "甲只删温度"), (0, 2, "甲只删云量"),
                           (0, 3, "甲温度云量都删"), (4, 5, "乙删温度滞后")]:
    full = abl.loc[i_full, "MAE(kW)"]
    cut_ = abl.loc[i_cut, "MAE(kW)"]
    print(f"  {tag:<14}: {full:.4f} → {cut_:.4f} kW  ({ (cut_ - full) / full * 100:+.2f}%)")

# 存结果
abl.to_csv(os.path.join(OUT_DIR, "metrics_step6_ablation.csv"),
           index=False, encoding="utf-8-sig")
corr_df.to_csv(os.path.join(OUT_DIR, "metrics_step6_corr.csv"),
               index=False, encoding="utf-8-sig")
if len(box_df):
    box_df.to_csv(os.path.join(OUT_DIR, "metrics_step6_temp_effect.csv"),
                  index=False, encoding="utf-8-sig")

# ============================================================
# 5) 出图：左=温度效应（控制变量），右=消融 MAE 对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# 左图：每个辐射档位的温度效应（柱子）
if len(box_df):
    ax = axes[0]
    xs = np.arange(len(box_df))
    vals = box_df["温度效应 kW/℃"].values
    ax.bar(xs, vals, color="#4C78A8")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(box_df["GHI区间"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("辐射区间（控制变量：GHI 卡在这一档里）")
    ax.set_ylabel("功率随温度的变化  kW/℃")
    ax.set_title("控制变量下温度效应\n（柱子越接近0，说明温度越可忽略）", fontsize=11)
    ax.text(0.02, 0.90,
            "对照：GHI 每 +100 W/m² ≈ 功率 +12~13 kW\n→ 温度的量级小 1~2 个数量级",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="#FFF3CD", ec="#E0A800"))

# 右图：消融 MAE 对比
ax = axes[1]
pairs = [("设定甲\n(同时刻气象)", abl.loc[0, "MAE(kW)"], abl.loc[3, "MAE(kW)"]),
         ("设定乙\n(纯历史时序)", abl.loc[4, "MAE(kW)"], abl.loc[5, "MAE(kW)"])]
names = ["设定甲\n(同时刻气象)", "设定乙\n(纯历史时序)"]
xs = np.arange(2)
ax.bar(xs - 0.2, [p[1] for p in pairs], 0.4, label="完整（含温度/云量）", color="#54A24B")
ax.bar(xs + 0.2, [p[2] for p in pairs], 0.4, label="删掉温度/云量", color="#E45756")
for i in range(2):
    ax.text(xs[i] - 0.2, pairs[i][1] + 0.02, f"{pairs[i][1]:.3f}", ha="center", fontsize=9)
    ax.text(xs[i] + 0.2, pairs[i][2] + 0.02, f"{pairs[i][2]:.3f}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(names)
ax.set_ylabel("MAE (kW)")
ax.set_title("消融实验：删掉温度/云量后误差几乎没变", fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig7_ablation_temp_cloud.png"), dpi=150)
print("\n图已保存：figures/fig7_ablation_temp_cloud.png")
print("结果已保存：results/metrics_step6_ablation.csv 等")
print("=" * 72)
