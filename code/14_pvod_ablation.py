# -*- coding: utf-8 -*-
"""
14_pvod_ablation.py
===================
特征消融（拓展二的补充 + 时间特征工程的量化，拓展②）：
整组删掉某一类特征，看误差恶化多少，以此判断这组特征有没有独立贡献。

PVOD 没有「云层不透明度」，消融对象换成气象里的次要因子：
  设定甲：删气温 → 再删气压与风速
  设定乙：删气温滞后 → 删时刻编码（检验时间特征工程值多少）→ 只留上一刻功率

⚠️ 删掉后误差几乎不变，并不等于该因素对光伏没有物理影响，
   只说明它的信息被别的特征（主要是辐射）吸收了。报告里必须写清楚。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvod_common import (RES, STATIONS, RANDOM_STATE, TIME_FEATS, GHI,
                         LMD_FEATS, load_station, add_time_features, build_b,
                         split_by_time, safe_lag)

PLAN_A = {
    "完整（10 个特征）": ["lmd_totalirrad", "lmd_diffuseirrad", "lmd_temperature",
                         "lmd_pressure", "lmd_windspeed", "lmd_winddirection"] + TIME_FEATS,
    "删气温（9 个）": ["lmd_totalirrad", "lmd_diffuseirrad",
                      "lmd_pressure", "lmd_windspeed", "lmd_winddirection"] + TIME_FEATS,
    "删气温+气压+风速（7 个）": ["lmd_totalirrad", "lmd_diffuseirrad",
                                "lmd_winddirection"] + TIME_FEATS,
}

LAG_B = ["GHI_lag1", "GHI_lag2", "GHI_lag3", "GHI_lag6", "GHI_lag12"]
PLAN_B = {
    "完整（12 个特征）": LAG_B + ["P_lag1", "T_lag1", "T_lag6"] + TIME_FEATS,
    "删气温滞后（10 个）": LAG_B + ["P_lag1"] + TIME_FEATS,
    "删时刻编码（8 个）": LAG_B + ["P_lag1", "T_lag1", "T_lag6"],
    "只留上一刻功率（5 个）": ["P_lag1"] + TIME_FEATS,
}

# 设定丙：回答「已知上一刻功率之后，同时刻气象还能不能再贡献一点」。
# 完整丙 = 6 个同时刻气象 + 上一刻功率/辐射 + 时刻编码；
# 逐层往下删，看删到只剩 Persistence 那一项时误差恶化多少。
PLAN_C = {
    "完整（12 个特征）": LMD_FEATS + ["P_lag1", "GHI_lag1"] + TIME_FEATS,
    "删全部同时刻气象（6 个）": ["P_lag1", "GHI_lag1"] + TIME_FEATS,
    "只留上一刻功率（5 个）": ["P_lag1"] + TIME_FEATS,
}


def run_one(d, feats):
    dd = d[feats + ["p_norm"]].dropna()
    tr, te = split_by_time(dd)
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(tr[feats].values, tr["p_norm"].values)
    return mean_absolute_error(te["p_norm"].values, rf.predict(te[feats].values)) * 100


def main():
    rows = []
    for sid in STATIONS:
        d = load_station(sid)
        da = add_time_features(d.copy())
        db, _ = build_b(d)
        dc = add_time_features(d.copy())
        safe_lag(dc, "p_norm", 1, "P_lag1")
        safe_lag(dc, GHI, 1, "GHI_lag1")

        for tag, plan, src in [("甲-同时刻实测(lmd)", PLAN_A, da),
                               ("乙-纯历史时序", PLAN_B, db),
                               ("丙-同时刻气象+历史", PLAN_C, dc)]:
            base = None
            for name, feats in plan.items():
                mae = run_one(src, feats)
                if base is None:
                    base = mae
                rows.append(dict(station=sid, 设定=tag, 方案=name, 特征数=len(feats),
                                 MAE=round(mae, 4),
                                 相对完整变化=round((mae - base) / base * 100, 2)))

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(RES, "pvod_ablation_per_station.csv"), index=False, encoding="utf-8-sig")
    # 十站平均：MAE 取平均，但「相对完整变化」也必须按平均之后的 MAE 再算一次，
    # 否则同一张表里 MAE 和百分比是两套口径，读者照着 MAE 手算会对不上百分比。
    # （逐站先算比值再平均，会被小分母的站点放大，丙设定上能差出 19 个百分点。）
    avg = (per.groupby(["设定", "方案", "特征数"], as_index=False)
              .agg(MAE=("MAE", "mean")))
    base = avg[avg["方案"].str.contains("完整")][["设定", "MAE"]]
    avg = (avg.merge(base, on="设定", suffixes=("", "_完整")))
    avg["相对完整变化"] = (avg["MAE"] - avg["MAE_完整"]) / avg["MAE_完整"] * 100
    avg = avg[["设定", "方案", "特征数", "MAE", "相对完整变化"]].round(4)
    avg.to_csv(os.path.join(RES, "pvod_ablation.csv"), index=False, encoding="utf-8-sig")

    print("=" * 82)
    print("特征消融（十站平均，随机森林，MAE 单位 %装机容量）")
    print(avg.to_string(index=False))
    print("\n已保存：results/pvod_ablation.csv 与 results/pvod_ablation_per_station.csv")


if __name__ == "__main__":
    main()
