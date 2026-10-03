# -*- coding: utf-8 -*-
"""
12_pvod_weather_scenarios.py
============================
天气分场景评估（拓展一）：把测试集按天气条件分组，看模型增益集中在哪类天气。

PVOD 没有「云量」字段，所以晴阴要靠晴空指数 kt 判定：

    kt = 实测总辐射 GHI / 大气层外水平面辐射 I0h

kt 越接近 1 说明越晴，越小说明云越厚。I0h 由各站自己的经纬度算太阳位置得到，
不是拍脑袋给的常数（见 solar_geometry）。

三个维度：
  ① 晴度     kt 分三档：阴 <0.35、多云 0.35~0.6、晴 ≥0.6
  ② 辐射突变 |GHI(t)-GHI(t-1)| 分四档：<10、10~30、30~60、>60 W/m²
  ③ 光照强度 GHI 分三档：<200、200~600、≥600 W/m²

比较对象：设定乙下 随机森林 相对 Persistence 基线的 MAE 降幅（十站测试集合并）。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvod_common import (RES, STATIONS, RANDOM_STATE, GHI, load_station,
                         build_b, split_by_time)


def solar_geometry(dt_index, lat, lon):
    """大气层外水平面辐射 I0h（W/m²）。
    赤纬用 Cooper 近似，时角按当地经度相对东八区标准子午线 120°E 修正。"""
    n = dt_index.dt.dayofyear.values.astype(float)
    delta = np.deg2rad(23.45 * np.sin(np.deg2rad(360 * (284 + n) / 365)))
    phi = np.deg2rad(lat)
    # 输入的时间戳已转成北京时间（见 pvod_common.load_station），
    # 东八区标准子午线 120°E，故按 (lon-120)/15 小时修正到地方真太阳时。
    hour_dec = dt_index.dt.hour.values + dt_index.dt.minute.values / 60.0
    solar_time = hour_dec + (lon - 120.0) / 15.0
    omega = np.deg2rad(15.0 * (solar_time - 12.0))
    sin_h = np.clip(np.sin(phi) * np.sin(delta)
                    + np.cos(phi) * np.cos(delta) * np.cos(omega), 0.0, None)
    I0 = 1367.0 * (1 + 0.033 * np.cos(np.deg2rad(360 * n / 365)))
    return I0 * sin_h, sin_h


MIN_SIN_H = 0.15   # 太阳高度角正弦下限，约 8.6°，低于此值不参与晴度分档
MIN_GHI = 50.0     # 有效发电门槛（W/m²），低于此值视为几乎没有出力


def bin_kt(v):
    return "阴 (kt<0.35)" if v < 0.35 else ("多云 (0.35≤kt<0.6)" if v < 0.60 else "晴 (kt≥0.6)")


def bin_ramp(v):
    if v < 10:
        return "平稳 <10"
    if v < 30:
        return "缓变 10~30"
    if v < 60:
        return "突变 30~60"
    return "剧变 >60"


def bin_ghi(v):
    return "弱光 <200" if v < 200 else ("中光 200~600" if v < 600 else "强光 ≥600")


def main():
    meta = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "data", "PVOD_hebei", "metadata.csv")).set_index("Station_ID")
    frames = []

    for sid in STATIONS:
        lat, lon = meta.loc[sid, "Latitude"], meta.loc[sid, "Longitude"]
        d = load_station(sid)
        d, feats = build_b(d)
        d = d.dropna(subset=feats + ["p_norm"]).reset_index(drop=True)

        I0h, sin_h = solar_geometry(d["date_time"], lat, lon)
        d["I0h"], d["sin_h"] = I0h, sin_h
        # 只在太阳有一定高度时谈晴阴：日出日落附近大气路径极长，
        # kt 天然很低，会把「阴」这一档撑得很大，不是云造成的。
        ok = (sin_h > MIN_SIN_H) & (I0h > 50)
        d["kt"] = np.where(ok, d[GHI] / np.where(I0h > 0, I0h, np.nan), np.nan)
        d["ramp"] = (d[GHI] - d[GHI].shift(1)).abs()
        d = d.dropna(subset=["kt"]).reset_index(drop=True)

        tr, te = split_by_time(d)
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(tr[feats].values, tr["p_norm"].values)
        pred = rf.predict(te[feats].values)

        te = te.copy()
        te["ae_rf"] = (te["p_norm"] - pred).abs() * 100
        te["ae_base"] = (te["p_norm"] - te["P_lag1"]).abs() * 100
        te["晴度"] = te["kt"].apply(bin_kt)
        te["辐射突变"] = te["ramp"].apply(bin_ramp)
        te["光照强度"] = te[GHI].apply(bin_ghi)
        te["station"] = sid
        frames.append(te[["station", "晴度", "辐射突变", "光照强度",
                          "ae_rf", "ae_base", GHI, "p_norm"]])

    all_te = pd.concat(frames, ignore_index=True)

    # ---- 两套口径 ----
    # ① 全部有光照样本（太阳高度角已过滤）：含大量接近零的低功率时刻
    # ② 有效发电时段（再要求实测辐射 > 50 W/m²）：
    #    否则「功率≈0、上一刻≈0」的样本会让 Persistence 近乎完美，
    #    MAE 分母极小，任何模型的相对降幅都会被人为放大成负几百个百分点。
    masks = {"全天有光照": np.ones(len(all_te), dtype=bool),
             "有效发电(辐射>50)": (all_te[GHI] > MIN_GHI).values}

    rows = []
    for scope, mk in masks.items():
        sub_all = all_te[mk]
        for dim in ["晴度", "辐射突变", "光照强度"]:
            for g, sub in sub_all.groupby(dim):
                m_rf, m_b = sub["ae_rf"].mean(), sub["ae_base"].mean()
                rows.append(dict(口径=scope, 维度=dim, 分组=g, 样本数=len(sub),
                                 RF_MAE=round(m_rf, 4), 基线_MAE=round(m_b, 4),
                                 降幅=round((m_b - m_rf) / m_b * 100, 2)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "pvod_weather_scenarios.csv"), index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("天气分场景（十站测试集合并，设定乙：RF vs Persistence，MAE 单位 %装机容量）")
    print(out.to_string(index=False))
    print("\n已保存：results/pvod_weather_scenarios.csv")


if __name__ == "__main__":
    main()
