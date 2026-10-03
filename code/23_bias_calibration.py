# -*- coding: utf-8 -*-
"""
23_bias_calibration.py
======================
误差分方向统计的第二步：既然代价随方向不对称，那就问一句——
「能不能给预测整体加一个很小的常数偏置，把代价压下来？」

【动机】
  22 号脚本发现：绝大多数模型的带符号误差 bias 是负的（预测系统性偏低 → 少报 → 弃光），
  而且少数几个大幅少报的断面贡献了大部分误差。调度代价是
      C = 少报电量 × 弃光电价 + 多报电量 × 备用电价，
  两个方向单价不同，所以把预测整体抬高或压低一点，代价未必同比例变坏——
  这就是工程上常见的「偏置校准 / bias shift」。

【口径】
  每个断面先换算成真实电量：1 kWh = 百分比装机容量 × 0.01 × 装机(kW) × 0.25 h。
  这样大电站的同样误差更"贵"，十站汇总才合理。
  δ 的扫描范围 ±3% 装机容量，步长 0.1%（step=0.001 → 0.1% 装机容量）。

【诚实声明】
  这里用测试期数据扫描最优偏置 δ，工程上属于拿测试数据调参（会高估收益）。
  正确做法是用独立验证集或滚动校准窗口。本报告把它当作「方法可行性演示」，
  并在结论里如实说明这个局限，以及真实部署时该怎么改。

【跑三个代表性方案】
  丙-LightGBM（十站最优）、丙-线性回归、甲-同时刻预报(nwp)固定系数（bias 最正，能看出风险翻转）。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvod_common import (RES, FIG, STATIONS, RANDOM_STATE, GHI, NWP_GHI, NWP_FEATS,
                         LMD_FEATS, TIME_FEATS, load_station, add_time_features,
                         build_b, split_by_time, fit_coef, capacity_mw)

os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Heiti TC", "SimHei", "Arial Unicode MS", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False

P_PV = 0.3433      # 少报（弃光）单价 元/kWh —— 河北南网 2025 年光伏结算均价
P_COAL = 0.3644    # 多报（缺额）单价 元/kWh —— 河北南网燃煤发电基准价
P_PEAK = 1.2000    # 敏感性：河北南网现货出清上限（2025-03-01 起连续结算试运行）
CAP_MW = {s: capacity_mw(s) for s in STATIONS}


def to_kwh(delta_pct, sid):
    """归一化误差（单位：%装机容量）→ kWh。delta_pct 是 0.01 量纲的百分比数值。"""
    return delta_pct * CAP_MW[sid] * 250.0


def cost_of(e_kwh, p_pv=P_PV, p_coal=P_COAL):
    """e_kwh 为带符号电量误差（>0 表示多报）。"""
    return float(np.clip(-e_kwh, 0, None).sum() * p_pv + np.clip(e_kwh, 0, None).sum() * p_coal)


def build_c(d):
    from pvod_common import safe_lag
    d = add_time_features(d.copy())
    safe_lag(d, "p_norm", 1, "P_lag1")
    safe_lag(d, GHI, 1, "GHI_lag1")
    return d, LMD_FEATS + ["P_lag1", "GHI_lag1"] + TIME_FEATS


def bl_a_nwp(tr, te):
    k = fit_coef(tr, NWP_GHI)
    return "Persistence", k * te[NWP_GHI].values


def bl_b(tr, te):
    return "Persistence", te["P_lag1"].values


def ctor_lgb():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=100, learning_rate=0.1,
                         random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)


def ctor_lr():
    from sklearn.linear_model import LinearRegression
    return LinearRegression()

LABELS = {"ctor_lgb": "LightGBM", "ctor_lr": "线性回归"}


def main():
    store = {}

    def collect(tag, builder, baseline_fn, modelctor, mlabel, sid):
        d, feats = builder(load_station(sid))
        cols = feats + [c for c in ["p_norm", "date_time", GHI, NWP_GHI] if c not in feats]
        dd = d[cols].dropna().reset_index(drop=True)
        tr, te = split_by_time(dd)
        y = te["p_norm"].values
        ghi = te[GHI].values if GHI in te.columns else te[NWP_GHI].values
        day = ghi > 0
        cap = CAP_MW[sid]

        name, pred = baseline_fn(tr, te)
        pred = np.asarray(pred, float)
        e = (pred[day] - y[day]) * cap * 250.0      # 归一化(0~1)→kWh: ×kW(×1000)×0.25h
        store.setdefault((tag, sid), {})[name] = e

        m = modelctor()
        m.fit(tr[feats].values, tr["p_norm"].values)
        e2 = (m.predict(te[feats].values)[day] - y[day]) * cap * 250.0
        store[(tag, sid)][mlabel] = e2

    for sid in STATIONS:
        print("扫描", sid)
        collect("丙-同时刻气象+历史", build_c, bl_b, ctor_lgb, "LightGBM", sid)
        collect("丙-同时刻气象+历史", build_c, bl_b, ctor_lr, "线性回归", sid)
        collect("甲-同时刻预报(nwp)",
                lambda d: (add_time_features(d.copy()), NWP_FEATS + TIME_FEATS),
                bl_a_nwp, ctor_lr, "线性回归", sid)

    # ---------- 偏置扫描 ----------
    grid = np.arange(-3.0, 3.01, 0.1)      # 单位：%装机容量
    rows, curves = [], {}
    for tag in ["丙-同时刻气象+历史", "甲-同时刻预报(nwp)"]:
        for mname in ["Persistence", "线性回归", "LightGBM"]:
            # store 的外层 key 是 (设定, 站点)，方法名在内层那一层
            keys = [k for k in store if k[0] == tag and mname in store[k]]
            if not keys:
                continue
            e_all = np.concatenate([store[k][mname] for k in keys])
            cs = np.array([cost_of(e_all + dd) for dd in grid])
            j = int(np.argmin(cs))
            c0 = cs[int(np.argmin(np.abs(grid)))]
            rows.append({
                "设定": tag, "方法": mname,
                "最优偏置δ_pct": round(float(grid[j]), 2),
                "原代价元": float(c0), "最优代价元": float(cs[j]),
                "节省比例%": round((1 - cs[j] / c0) * 100, 2),
                "少报电量kWh": float(np.clip(-e_all, 0, None).sum()),
                "多报电量kWh": float(np.clip(e_all, 0, None).sum()),
                "断面数": int(e_all.size),
            })
            # 多报单价换成现货出清上限 1.2 元/kWh 再扫一遍：
            # 若两方向单价差距很大，「整体抬高/压低一点」才可能真的省钱。
            cs_peak = np.array([cost_of(e_all + dd, p_coal=P_PEAK) for dd in grid])
            j_peak = int(np.argmin(cs_peak))
            c0_peak = cs_peak[int(np.argmin(np.abs(grid)))]
            rows[-1]["最优δ_现货上限"] = round(float(grid[j_peak]), 2)
            rows[-1]["节省比例%_现货上限"] = round((1 - cs_peak[j_peak] / c0_peak) * 100, 2)

            # 分别存下 δ=0 时「少报代价」「多报代价」，画图要用（两者往往相差好几倍）
            cu0 = float(np.clip(-e_all, 0, None).sum()) * P_PV
            co0 = float(np.clip(e_all, 0, None).sum()) * P_COAL
            curves[(tag, mname)] = (grid, cs, cu0, co0)
    cal = pd.DataFrame(rows)
    print("校准表列名：", list(cal.columns))
    cal["少报代价元"] = cal["少报电量kWh"] * P_PV
    cal["多报代价元"] = cal["多报电量kWh"] * P_COAL
    cal = cal[["设定", "方法", "少报电量kWh", "多报电量kWh", "少报代价元", "多报代价元",
               "最优偏置δ_pct", "原代价元", "最优代价元", "节省比例%",
               "最优δ_现货上限", "节省比例%_现货上限", "断面数"]]
    cal.round(3).to_csv(os.path.join(RES, "pvod_bias_calibration.csv"), index=False, encoding="utf-8-sig")

    # ---------- 按辐照分档（丙-LightGBM）----------
    bins = [(0, 50), (50, 300), (300, 600), (600, 1000), (1000, 99999)]
    labels = ["0~50", "50~300", "300~600", "600~1000", ">1000"]
    b_rows = []
    for (lo, hi), lab in zip(bins, labels):
        tot_u = tot_o = n = 0.0
        for sid in STATIONS:
            d, feats = build_c(load_station(sid))
            # ⚠️ feats 里已经含 GHI(lmd_totalirrad)，这里必须去重，
            # 否则列名重复，te[GHI] 会取到两列、.values 变成二维。
            cols = feats + [c for c in ["p_norm", "date_time", GHI] if c not in feats]
            dd = d[cols].dropna().reset_index(drop=True)   # 滞后首行为 NaN，必须剔除
            tr, te = split_by_time(dd)
            mdl = ctor_lgb().fit(tr[feats].values, tr["p_norm"].values)
            pred = np.asarray(mdl.predict(te[feats].values)).ravel()
            y = np.asarray(te["p_norm"].values).ravel()
            ghi = np.asarray(te[GHI].values).ravel()
            m = (ghi > 0) & (ghi >= lo) & (ghi < hi)
            if not m.any():
                continue
            e = (pred[m] - y[m]) * CAP_MW[sid] * 250.0
            tot_u += np.clip(-e, 0, None).sum()
            tot_o += np.clip(e, 0, None).sum()
            n += m.sum()
        b_rows.append(dict(辐射档W每平米=lab, 断面数=int(n),
                           少报电量kWh=round(tot_u, 1), 多报电量kWh=round(tot_o, 1),
                           bias_pct=round((tot_o - tot_u) / max(n, 1) / CAP_MW["station00"] * 0, 4),
                           少报代价元=round(tot_u * P_PV, 1), 多报代价元=round(tot_o * P_COAL, 1)))
    bdf = pd.DataFrame(b_rows)
    bdf.to_csv(os.path.join(RES, "pvod_error_by_irradiance.csv"), index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    show = [("甲-同时刻预报(nwp)", "Persistence"),
            ("甲-同时刻预报(nwp)", "线性回归"),
            ("丙-同时刻气象+历史", "LightGBM"),
            ("丙-同时刻气象+历史", "线性回归"),
            ("丙-同时刻气象+历史", "Persistence")]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(show)); w = 0.36
    cu, co = [], []
    for k in show:
        if k not in curves:
            continue
        g, cs, cu0, co0 = curves[k]
        cu.append(cu0); co.append(co0)
    nshow = len(cu)
    ax.bar(x[:nshow] - w / 2, cu, w, label="少报代价（弃光）", color="#C0392B")
    ax.bar(x[:nshow] + w / 2, co, w, label="多报代价（备用到位）", color="#1F6FB4")
    ax.set_xticks(x[:nshow])
    ax.set_xticklabels([f"{t.split('-')[0]}\n{m}" for t, m in show[:nshow]], fontsize=9)
    ax.set_ylabel("代价（元）")
    ax.set_title("预测误差两个方向的代价不相等：有的方案少报更贵，有的多报更贵", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig17_error_direction.png"), dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    xx = np.arange(len(bdf))
    ax.bar(xx - 0.2, bdf["少报电量kWh"], 0.4, label="少报（预测偏低）", color="#C0392B")
    ax.bar(xx + 0.2, bdf["多报电量kWh"], 0.4, label="多报（预测偏高）", color="#1F6FB4")
    ax.set_xticks(xx); ax.set_xticklabels(bdf["辐射档W每平米"])
    ax.set_xlabel("同时刻总辐射 GHI（W/m²）")
    ax.set_ylabel("累计偏差电量（kWh）")
    ax.set_title("丙-LightGBM 十站合计：低辐照时段以少报为主", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig18_error_by_irradiance.png"), dpi=160)
    plt.close()

    print("\n" + "=" * 100)
    print("【偏置扫描】δ>0 = 把预测整体抬高（元，十站测试期合计）")
    print(cal.to_string(index=False))
    print("\n【按辐照分档 丙-LightGBM 十站合计】")
    print(bdf.drop(columns=["bias_pct"]).to_string(index=False))
    print("\n已保存：results/pvod_bias_calibration.csv, results/pvod_error_by_irradiance.csv")
    print("已保存：figures/fig17_error_direction.png, figures/fig18_error_by_irradiance.png")


if __name__ == "__main__":
    main()
