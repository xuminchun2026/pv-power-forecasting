# -*- coding: utf-8 -*-
"""
============================================================
第 5 步：两项拓展
  拓展① 天气分场景评估（误差到底躲在哪种天气里）
  拓展② 特征重要性可解释性（谁才是真正说了算的特征）
============================================================
怎么运行：PyCharm 里右键本文件 -> 运行 '04_interpretability'

【先记住第 4 步的结论】
  设定乙（只能用历史观测）里：RF 的 MAE = 0.6681，上一时刻基线 = 1.5374
  → 机器学习把误差砍了 56.5%。
  但！**整体平均一刀切，会掩盖一件事**：是不是所有时刻都降了一半？
  很可能大部分时刻降 80%，少数阴天反而变糟 —— 平均数字把它糊住了。
  这一步就是要把 10501 个测试样本**按天气拆开**，看误差长什么样。

【⚠️ 一个必须说清的"不算泄露"】
  分场景时我们会用到"当前时刻的真实 GHI"。
  但它【只用于事后分组统计】，模型训练时根本没见过它（设定乙只用了 lag 特征）。
  所以模型预测是干净的、公平的；分场景只是"考完试再看卷子分类统计"。
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# 中文字体：macOS 用 Arial Unicode MS，否则中文显示成方块
# ---------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = "/Users/xuminchun/Desktop/电气club作业/data/ningxia_2019_15min.xlsx"
TIME_COL = "采样结束时刻"
TARGET = "实际功率"
GHI = "GHI（太阳总水平辐射）"
DNI = "DNI（太阳直接辐射指数）"
RANDOM_STATE = 42

FIG_DIR = "/Users/xuminchun/Desktop/电气club作业/figures"
OUT_DIR = "/Users/xuminchun/Desktop/电气club作业/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================
# 构建设定乙（和第 4 步完全一样，这里复制一份，让本文件能独立跑）
# ===========================================================
def build_setting_b(df):
    d = df.copy().sort_values(TIME_COL).reset_index(drop=True)
    t = d[TIME_COL]
    d["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    d["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365)
    d["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365)

    d["GHI_lag1"] = d[GHI].shift(1)
    d["GHI_lag2"] = d[GHI].shift(2)
    d["GHI_lag3"] = d[GHI].shift(3)
    d["GHI_lag6"] = d[GHI].shift(6)
    d["GHI_lag12"] = d[GHI].shift(12)
    d["P_lag1"] = d[TARGET].shift(1)
    d["T_lag1"] = d["气温"].shift(1)
    d["T_lag6"] = d["气温"].shift(6)

    feats = ["GHI_lag1", "GHI_lag2", "GHI_lag3", "GHI_lag6", "GHI_lag12",
             "P_lag1", "T_lag1", "T_lag6",
             "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    # 注意：GHI/DNI/气温 的"当前时刻值"【不进 feats】（那是泄露），
    # 保留它们只是为了第 5 步【事后分场景统计】用。
    d = d[feats + [TARGET, TIME_COL, GHI, DNI, "气温"]].dropna()
    return d, feats


# ===========================================================
# 主流程
# ===========================================================
df = pd.read_excel(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
data, feats = build_setting_b(df)

cut = int(len(data) * 0.7)
train, test = data.iloc[:cut], data.iloc[cut:]

rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(train[feats].values, train[TARGET].values)

test = test.copy()
test["RF预测"] = rf.predict(test[feats].values)
test["基线预测(上一时刻功率)"] = test["P_lag1"].values
test["真实功率"] = test[TARGET].values
test["RF误差"] = (test["RF预测"] - test["真实功率"]).abs()
test["基线误差"] = (test["基线预测(上一时刻功率)"] - test["真实功率"]).abs()

print("=" * 72)
print(f"测试集 {len(test)} 条 | 训练集只用前 70%，没打乱（防数据泄露）")

# 只看白天（功率>0），否则夜间的 0 误差会把统计淹没
day = test[test[TARGET] > 0].copy()
print(f"其中白天发电时段 {len(day)} 条，下面所有分析都基于这 {len(day)} 条")

# ----------------------------------------------------------
# 拓展① 三种天气分场景
# ----------------------------------------------------------
print("=" * 72)
print("【拓展①】天气分场景：误差到底躲在哪种天气里？")

# 场景一：按"晴度"（DNI/GHI）。太阳直接光占比高=晴，低=阴
day["晴度"] = day[DNI] / day[GHI].replace(0, np.nan)
day["晴度档"] = pd.cut(day["晴度"], [0, 0.30, 0.50, 0.70, 0.85, 2.0],
                       labels=["很阴 <0.30", "偏阴 0.30-0.50",
                               "半晴 0.50-0.70", "晴 0.70-0.85", "很晴 >0.85"])

# 场景二：按"辐射突变"（一刻钟内 GHI 跳多少）—— 跳得越猛越可能是云团跑过
day["辐射突变"] = (day[GHI] - day["GHI_lag1"]).abs()
day["突变档"] = pd.cut(day["辐射突变"], [-1, 10, 30, 60, 10000],
                       labels=["平静 <10", "一般 10-30",
                               "剧烈 30-60", "狂暴 >60"])

# 场景三：按"光照强度"
day["强度档"] = pd.cut(day[GHI], [-1, 200, 600, 10000],
                       labels=["弱光 <200", "中光 200-600", "强光 >600"])


def summarise(col):
    g = day.groupby(col, observed=True).agg(
        样本数=("RF误差", "size"),
        RF_MAE=("RF误差", "mean"),
        基线MAE=("基线误差", "mean"))
    g["误差降幅%"] = 100 * (1 - g["RF_MAE"] / g["基线MAE"])
    return g.round(4)


tables = {}
for name, col in [("按晴度 DNI/GHI", "晴度档"),
                  ("按辐射突变 |ΔGHI|", "突变档"),
                  ("按光照强度 GHI", "强度档")]:
    t = summarise(col)
    tables[name] = t
    print(f"\n--- {name} ---")
    print(t.to_string())

all_scene = pd.concat(
    [t.assign(分场景维度=n) for n, t in tables.items()]).reset_index()
all_scene.to_csv(os.path.join(OUT_DIR, "metrics_step5_weather.csv"),
                 index=False, encoding="utf-8-sig")

# ----------------------------------------------------------
# 拓展② 特征重要性
# ----------------------------------------------------------
print("=" * 72)
print("【拓展②】特征重要性：谁才是说了算的？")

imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
print("\n① 随机森林自带的特征重要性（越大越重要）：")
print(imp.round(6).to_string())

print("\n② 排列重要性（把某列打乱，看误差涨多少 —— 比①更可信）：")
pi = permutation_importance(rf, day[feats].values, day[TARGET].values,
                            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
imp2 = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
print(imp2.round(6).to_string())
print("（打乱一列后 MAE 平均涨多少 kW，涨得越多说明这列越关键）")

print("\n③ 留一法：把某一列整个删掉重新训练，看 MAE 变成多少：")
base_mae = mean_absolute_error(day[TARGET], day["RF预测"])
print(f"   完整特征时 MAE = {base_mae:.4f} kW")
drop_rows = []
for f in feats:
    rest = [c for c in feats if c != f]
    m = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(train[rest].values, train[TARGET].values)
    mae = mean_absolute_error(day[TARGET], m.predict(day[rest].values))
    drop_rows.append({"被删掉的列": f, "MAE(kW)": round(mae, 4),
                      "比完整差多少": round(mae - base_mae, 4)})
drop_res = pd.DataFrame(drop_rows).sort_values("比完整差多少", ascending=False)
print(drop_res.to_string(index=False))

pd.concat([imp.rename("内置重要性"), imp2.rename("排列重要性")], axis=1) \
    .sort_values("排列重要性", ascending=False).to_csv(
    os.path.join(OUT_DIR, "metrics_step5_importance.csv"),
    encoding="utf-8-sig")

# ----------------------------------------------------------
# 画图
# ----------------------------------------------------------
print("=" * 72)
print("画图…")

# 图5 天气分场景
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, t) in zip(axes, tables.items()):
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["基线MAE"], 0.4, label="上一时刻基线", color="#8fb8de")
    ax.bar(x + 0.2, t["RF_MAE"], 0.4, label="随机森林RF", color="#d95f5f")
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MAE (kW)")
    ax.set_title(name)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("图5  天气分场景下的预测误差（越低越好）", fontsize=14)
plt.tight_layout()
p5 = os.path.join(FIG_DIR, "fig5_weather_scenarios.png")
plt.savefig(p5, dpi=150, bbox_inches="tight")
plt.close()

# 图6 特征重要性
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
imp.sort_values().plot.barh(ax=axes[0], color="#4e79a7")
axes[0].set_title("图6a  随机森林内置重要性")
imp2.sort_values().plot.barh(ax=axes[1], color="#e15759")
axes[1].set_title("图6b  排列重要性（打乱该列后 MAE 的涨幅）")
for ax in axes:
    ax.set_xlabel("重要性")
    ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
p6 = os.path.join(FIG_DIR, "fig6_feature_importance.png")
plt.savefig(p6, dpi=150, bbox_inches="tight")
plt.close()

print(f"已保存：{p5}")
print(f"已保存：{p6}")
print("\n[OK] 第 5 步跑完。把输出发我，我们进第 6 步：写报告 + GitHub。")
