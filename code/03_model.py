# -*- coding: utf-8 -*-
"""
============================================================
第 4 步：建模 —— 两个预测设定 + 朴素基线 + 机器学习模型
============================================================
怎么运行：PyCharm 里右键本文件 -> 运行 '03_model'

【这一步要研究的问题】
预测电站"当前时刻"的输出功率。三个对比：
  1) 朴素基线（不用机器学习的土办法）能不能打？
  2) 随机森林 / 梯度提升树 能不能比基线更准？准多少？
  3) "同时刻气象信息" 和 "纯历史观测" 哪种预测能力更强？

【两个设定（本项目的研究设计）】
  设定甲 NOWCAST（同时刻气象）：
      特征 = 预测时刻"同时刻"就能测到的气象（GHI、气温、云量…）
      这是题目的本意：用气象预测功率。
  设定乙 HIST（纯历史时序 / 实时预测）：
      特征 = 只能拿到"预测时刻之前"的观测（上一刻、上两刻的辐射与功率…）
      这才是真正的时序预测：当前时刻的辐射要等它过去才知道，不能拿来当输入。
      现实里对应"用气象站的历史观测 + 短期预报前值"估算当前功率。

【红线提醒（学术诚信 · 答辩常考，务必理解）】
  1) 绝不能随机打乱后再切分训练/测试集 —— 那等于"用未来数据预测过去"，叫数据泄露。
     本脚本按【时间先后】切分，测试集永远晚于训练集。
  2) 如果跑出来"机器学习没比基线好"，那【就是真实结果，不许偷偷调参把它调上去】。
     如实报告的发现，比编一个好看的数字值钱得多。
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 0) 全局设置
# ============================================================
DATA_PATH = "/Users/xuminchun/Desktop/电气club作业/data/ningxia_2019_15min.xlsx"
TIME_COL = "采样结束时刻"
TARGET = "实际功率"
RANDOM_STATE = 42          # 固定随机种子：换个电脑跑出来结果一样（答辩可复现点）

GHI = "GHI（太阳总水平辐射）"
DNI = "DNI（太阳直接辐射指数）"
DHI = "DHI（太阳散射辐射指数）"
GTI = "GTI（固定倾角辐射）"
TEMP = "气温"
WIND = "高度10m风速"

OUT_DIR = "/Users/xuminchun/Desktop/电气club作业/results"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1) 时间特征（拓展③）
# ============================================================
def add_time_features(d):
    """
    把"几点 / 哪一天"变成两个数字：sin 和 cos。
    为什么不能直接把"小时数 0~23"塞给模型？
      因为 23 点和 0 点明明是邻居，数字上却差了 23。
      sin/cos 把它搬到圆周上，23:00 和 00:00 就挨在一起了。
    """
    t = d[TIME_COL]
    d["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    d["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365)
    d["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365)
    return d


# ============================================================
# 2) 构建设定甲：同时刻气象特征
# ============================================================
def build_setting_a(df):
    d = add_time_features(df.copy())
    feats = [GHI, DNI, DHI, GTI, TEMP, "相对湿度", "云层不透明度", WIND,
             "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    d = d[feats + [TARGET, TIME_COL]].dropna()
    return d, feats


# ============================================================
# 3) 构建设定乙：纯历史（滞后）特征
# ============================================================
def build_setting_b(df):
    """
    shift(k) = 往回挪 k 条 = 取 "k 个 15 分钟之前" 的旧值。
    这里绝不能出现预测时刻自己的 GHI —— 用了就是作弊（数据泄露）。
    """
    d = df.copy().sort_values(TIME_COL).reset_index(drop=True)
    d = add_time_features(d)

    d["GHI_lag1"] = d[GHI].shift(1)
    d["GHI_lag2"] = d[GHI].shift(2)
    d["GHI_lag3"] = d[GHI].shift(3)
    d["GHI_lag6"] = d[GHI].shift(6)     # 1.5 小时前
    d["GHI_lag12"] = d[GHI].shift(12)   # 3 小时前
    d["P_lag1"] = d[TARGET].shift(1)    # 上一刻功率（真实现场一定拿得到）
    d["T_lag1"] = d[TEMP].shift(1)
    d["T_lag6"] = d[TEMP].shift(6)

    feats = ["GHI_lag1", "GHI_lag2", "GHI_lag3", "GHI_lag6", "GHI_lag12",
             "P_lag1", "T_lag1", "T_lag6",
             "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    d = d[feats + [TARGET, TIME_COL]].dropna()
    return d, feats


# ============================================================
# 4) 评估与训练
# ============================================================
def evaluate(tag, y_true, pred, name):
    return {
        "设定": tag,
        "方法": name,
        "MAE(kW)": round(mean_absolute_error(y_true, pred), 4),
        "RMSE(kW)": round(np.sqrt(mean_squared_error(y_true, pred)), 4),
        "R2": round(r2_score(y_true, pred), 4),
    }


def make_models():
    """机器学习模型（随机种子固定，保证可复现）"""
    return {
        "线性回归": LinearRegression(),
        "随机森林RF": RandomForestRegressor(n_estimators=200,
                                            random_state=RANDOM_STATE, n_jobs=-1),
        "梯度提升树GBDT": GradientBoostingRegressor(n_estimators=100,
                                                    random_state=RANDOM_STATE),
    }


print("=" * 72)
print("载入数据…")
df = pd.read_excel(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)

data_a, feats_a = build_setting_a(df)
data_b, feats_b = build_setting_b(df)

n_a, n_b = len(data_a), len(data_b)
cut_a, cut_b = int(n_a * 0.7), int(n_b * 0.7)
tr_a, te_a = data_a.iloc[:cut_a], data_a.iloc[cut_a:]
tr_b, te_b = data_b.iloc[:cut_b], data_b.iloc[cut_b:]

print(f"设定甲：{n_a} 条 | 特征 {len(feats_a)} 个")
print(f"设定乙：{n_b} 条 | 特征 {len(feats_b)} 个")
print(f"切分点（第70%）：甲 {te_a[TIME_COL].min().date()} 起｜乙 {te_b[TIME_COL].min().date()} 起")
print("（测试集全部晚于训练集 —— 这就是防数据泄露的关键，绝不能打乱）")

all_rows = []

# ---------------- 设定甲 ----------------
print("\n" + "=" * 72)
print("【设定甲】同时刻气象 → 预测当前功率")

# 训练集里估出那个"固定系数"（用测试集算就泄露了）
coef_a = float(np.polyfit(tr_a[GHI], tr_a[TARGET], 1)[0])
base_a = (coef_a * te_a[GHI].values)
all_rows.append(evaluate("甲-同时刻气象", te_a[TARGET].values, base_a,
                         f"朴素基线: 功率≈{coef_a:.6f}×GHI"))

Xtr_a, ytr_a = tr_a[feats_a].values, tr_a[TARGET].values
Xte_a, yte_a = te_a[feats_a].values, te_a[TARGET].values
for name, mdl in make_models().items():
    mdl.fit(Xtr_a, ytr_a)
    all_rows.append(evaluate("甲-同时刻气象", yte_a, mdl.predict(Xte_a), name))

# ---------------- 设定乙 ----------------
print("=" * 72)
print("【设定乙】纯历史观测 → 预测当前功率（真正的时序预测）")

base_b = te_b["P_lag1"].values
all_rows.append(evaluate("乙-纯历史时序", te_b[TARGET].values, base_b,
                         "朴素基线: 上一时刻功率(Persistence)"))

Xtr_b, ytr_b = tr_b[feats_b].values, tr_b[TARGET].values
Xte_b, yte_b = te_b[feats_b].values, te_b[TARGET].values
for name, mdl in make_models().items():
    mdl.fit(Xtr_b, ytr_b)
    all_rows.append(evaluate("乙-纯历史时序", yte_b, mdl.predict(Xte_b), name))

# ---------------- 汇总 ----------------
res = pd.DataFrame(all_rows)
print("=" * 72)
print("【结果汇总】MAE/RMSE 越小越好，R2 越大越好")
print(res.to_string(index=False))

path = os.path.join(OUT_DIR, "metrics_step4.csv")
res.to_csv(path, index=False, encoding="utf-8-sig")
print(f"\n已保存到：{path}")

# ---------------- 追加：误差最大的时刻（为拓展①铺路）----------------
print("=" * 72)
print("【追加分析】设定乙里 RF 误差最大的 5 个时刻：")
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(Xtr_b, ytr_b)
pred_b = rf.predict(Xte_b)
err = pd.Series(np.abs(yte_b - pred_b), index=te_b.index).sort_values(ascending=False)
top_idx = err.head(5).index
cols = [TIME_COL, GHI, GTI, TEMP, "云层不透明度", TARGET]
show = df.loc[df.index.intersection(top_idx), cols].copy()
show["GHI前后差"] = show[GHI].diff().abs().round(0)
print(show.to_string())

print("\n[OK] 第 4 步跑完。看到结果后先别急着调参，把输出发给我。")
