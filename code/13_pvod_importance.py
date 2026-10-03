# -*- coding: utf-8 -*-
"""
13_pvod_importance.py
=====================
可解释性分析（拓展二）：输出特征重要性，回答「哪个气象因素最关键」。

做法：每站用随机森林跑设定甲与设定乙，取 feature_importances_，
再把十站的重要性取平均后排序 —— 单站排序有偶然性，十站平均才稳。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvod_common import (RES, STATIONS, RANDOM_STATE, NAME_CN, GHI,
                         load_station, add_time_features, build_b,
                         feats_a_lmd, feats_a_nwp, split_by_time, safe_lag,
                         LMD_FEATS, TIME_FEATS)


def build_c(d):
    """与 11 号脚本的设定丙保持一致：同时刻站内气象 + 上一刻功率与辐射。"""
    d = add_time_features(d.copy())
    safe_lag(d, "p_norm", 1, "P_lag1")
    safe_lag(d, GHI, 1, "GHI_lag1")
    return d, LMD_FEATS + ["P_lag1", "GHI_lag1"] + TIME_FEATS


def main():
    acc = {}
    for sid in STATIONS:
        d = load_station(sid)

        for tag, feats in [("甲-同时刻实测(lmd)", feats_a_lmd()),
                           ("甲-同时刻预报(nwp)", feats_a_nwp())]:
            da = add_time_features(d.copy())
            dd = da[feats + ["p_norm"]].dropna()
            tr, _ = split_by_time(dd)
            rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(tr[feats].values, tr["p_norm"].values)
            for f, v in zip(feats, rf.feature_importances_):
                acc[(tag, f)] = acc.get((tag, f), []) + [v]

        db, fb = build_b(d)
        dd = db[fb + ["p_norm"]].dropna()
        tr, _ = split_by_time(dd)
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(tr[fb].values, tr["p_norm"].values)
        for f, v in zip(fb, rf.feature_importances_):
            acc[("乙-纯历史时序", f)] = acc.get(("乙-纯历史时序", f), []) + [v]

        dc, fc = build_c(d)
        dd = dc[fc + ["p_norm"]].dropna()
        tr, _ = split_by_time(dd)
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(tr[fc].values, tr["p_norm"].values)
        for f, v in zip(fc, rf.feature_importances_):
            acc[("丙-同时刻气象+历史", f)] = acc.get(("丙-同时刻气象+历史", f), []) + [v]

    rows = [dict(设定=t, 特征=f, 中文名=NAME_CN.get(f, f),
                 重要性=round(float(np.mean(vs)), 5),
                 站间标准差=round(float(np.std(vs)), 5))
            for (t, f), vs in acc.items()]
    out = (pd.DataFrame(rows)
             .sort_values(["设定", "重要性"], ascending=[True, False])
             .reset_index(drop=True))
    out.to_csv(os.path.join(RES, "pvod_importance.csv"), index=False, encoding="utf-8-sig")

    print("=" * 78)
    print("特征重要性（十站平均，随机森林）")
    for tag in out["设定"].unique():
        print(f"\n---- {tag} ----")
        print(out[out["设定"] == tag][["中文名", "重要性", "站间标准差"]].to_string(index=False))
    print("\n已保存：results/pvod_importance.csv")


if __name__ == "__main__":
    main()
