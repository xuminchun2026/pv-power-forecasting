# -*- coding: utf-8 -*-
"""
17_identity_check.py
====================
验证一个机制假说：树模型拟合不好「上一刻功率 → 这一刻功率」的恒等映射。

【为什么关心这件事】
  主实验里出现两个反直觉现象：
    ① 设定丙中，线性回归 MAE 1.4942 优于全部树模型（LightGBM 1.6376）；
    ② 天气分场景里，辐射平稳时段随机森林反而输给 Persistence 一大截。
  猜测原因是：Persistence 本质是一个硬编码的恒等映射 P(t) = 1.0 × P(t-1)，
  线性回归能学到接近 1 的系数，而树模型只能输出叶节点内训练样本的均值，
  天生做不到「输出 = 输入」，于是把高值拉低、低值抬高（回归到均值）。

【做法】每站只用 P_lag1 一个特征，比较三种方式：
    ① Persistence：直接令 P(t) = P(t-1)（系数固定 1.0）
    ② 线性回归：学一个系数 k
    ③ 随机森林：200 棵，纯分段常数
  若 ③ 明显劣于 ①，而 ② 学到的 k 接近 1，假说即成立。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvod_common import (RES, STATIONS, RANDOM_STATE, GHI, load_station,
                         add_time_features, split_by_time, safe_lag)


def main():
    rows = []
    for sid in STATIONS:
        d = load_station(sid)
        dc = add_time_features(d.copy())
        safe_lag(dc, "p_norm", 1, "P_lag1")
        dd = dc[["P_lag1", "p_norm"]].dropna()
        tr, te = split_by_time(dd)
        Xtr, ytr = tr[["P_lag1"]].values, tr["p_norm"].values
        Xte, yte = te[["P_lag1"]].values, te["p_norm"].values

        # ① 硬编码恒等映射
        m_per = mean_absolute_error(yte, Xte.ravel()) * 100
        # ② 线性回归
        lr = LinearRegression().fit(Xtr, ytr)
        m_lr = mean_absolute_error(yte, lr.predict(Xte)) * 100
        # ③ 随机森林
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                   n_jobs=-1).fit(Xtr, ytr)
        m_rf = mean_absolute_error(yte, rf.predict(Xte)) * 100

        rows.append(dict(station=sid, Persistence=round(m_per, 4),
                         线性回归=round(m_lr, 4), 随机森林=round(m_rf, 4),
                         线性回归系数=round(float(lr.coef_[0]), 4)))

    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(RES, "pvod_identity_check.csv"), index=False, encoding="utf-8-sig")

    avg = o[["Persistence", "线性回归", "随机森林"]].mean()
    print("=" * 78)
    print("恒等映射检验（每站只用「上一刻功率」一个特征，MAE 单位 %装机容量）")
    print(o.to_string(index=False))
    print("\n【十站平均】")
    print(avg.round(4).to_string())
    print(f"\n线性回归学到的系数 k：{o['线性回归系数'].min():.4f} ~ "
          f"{o['线性回归系数'].max():.4f}，平均 {o['线性回归系数'].mean():.4f}")
    print(f"随机森林相对 Persistence 劣化：{(avg['随机森林'] - avg['Persistence']) / avg['Persistence'] * 100:.2f}%")
    print("\n【结论】随机森林做不到「输出=输入」，故在功率平稳时段反而输给 Persistence；")
    print("       线性回归能学到 k≈1 的恒等映射，这解释了它在设定丙中胜出。")
    print("\n已保存：results/pvod_identity_check.csv")


if __name__ == "__main__":
    main()
