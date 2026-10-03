# -*- coding: utf-8 -*-
"""
22_error_direction.py
=====================
误差的「分方向」统计 + 折算成工程代价。

【为什么单看 MAE 不够】
  MAE 把正误差和负误差一视同仁地取绝对值再平均，它只知道「差多少」，
  不知道「差在哪一边」。可电网调度里，两个方向的代价是不一样的：

    · 少报（预测 < 实际，e = 预测 - 真实 < 0）
      实际发的比预想的多，调度按偏低的预测留了裕度，多余的发电装不进去，
      只能弃光 → 损失的是「本来能卖钱的清洁电量」。
    · 多报（预测 > 真实，e > 0）
      实际发的比预想少，出现功率缺额，只能靠旋转备用 / 储能 / 快速启停机组顶，
      还要吃并网考核 → 损失的是「临时补电的钱」，单位代价通常更高。

  所以本报告把误差拆成正、负两侧分别统计（bias、两侧占比、两侧累计电量），
  再乘上真实电价折算成钱，看哪一侧更"贵"。

【电价锚点：全部来自公开文件，不瞎编】
  P_PV    0.3433 元/kWh  河北南网 2025 年光伏结算均价 343.29 元/MWh
                         （2025 年冀南光伏结算电量 83.96 亿 kWh 的加权均价）
  P_COAL  0.3644 元/kWh  河北南网燃煤发电基准价
                         （河北省发改委 136 号文实施方案：存量新能源机制电价按此执行）
  P_PEAK  1.2000 元/kWh  河北南网现货市场出清限价上限 1200 元/MWh
                         （2025-03-01 起连续结算试运行，申报/出清限价 0~1200）
  CAP_26  165 元/kW·年   河北省 2026 年起煤电容量电价（用来折算"必须留着备用"的年成本）

【口径声明】
  折算结果是「上界量级估算」，不是调度成本模型：
  假设每个断面的方向性偏差都全额兑现成弃电或缺额（实际调度有吸收能力），
  所以同时给出 0%/100% 兑现率下的区间。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvod_common import (BASE, RES, FIG, STATIONS, RANDOM_STATE, GHI, NWP_GHI,
                         TIME_FEATS, LMD_FEATS, load_station, add_time_features,
                         build_b, feats_a_lmd, feats_a_nwp, split_by_time,
                         fit_coef, capacity_mw)

os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

N_TREES = 100
LR_BOOST = 0.1

# ---------- 电价锚点（元 / kWh）----------
P_PV = 0.3433     # 少报：弃掉的电本来能卖这个价
P_COAL = 0.3644   # 多报：缺额得用火电顶，这度电的等效代价
P_PEAK = 1.2000   # 敏感性：现货出清上限
CAP_26 = 165.0    # 煤电容量电价（元/kW·年），2026 年起河北标准

PRICES = {"少报(弃光)基准": P_PV, "多报(缺额)基准": P_COAL, "多报(现货上限)": P_PEAK}


def make_models():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    models = {
        "线性回归": LinearRegression(),
        "随机森林RF": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(n_estimators=N_TREES, learning_rate=LR_BOOST,
                                           random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    except Exception:
        pass
    return models


def bl_a_lmd(tr, te):
    k = fit_coef(tr, GHI)
    return f"基线:固定系数×辐射", k * te[GHI].values


def bl_a_nwp(tr, te):
    k = fit_coef(tr, NWP_GHI)
    return f"基线:固定系数×辐射", k * te[NWP_GHI].values


def bl_b(tr, te):
    return "基线:Persistence", te["P_lag1"].values


def build_c(d):
    from pvod_common import safe_lag
    d = add_time_features(d.copy())
    safe_lag(d, "p_norm", 1, "P_lag1")
    safe_lag(d, GHI, 1, "GHI_lag1")
    feats = LMD_FEATS + ["P_lag1", "GHI_lag1"] + TIME_FEATS
    return d, feats


def direction_rows(sid, tag, rows):
    """rows: 每个断面一行，含 true/pred/ghi/hour"""
    e = rows["pred"].values - rows["true"].values        # <0 少报, >0 多报
    n = len(e)
    under = np.clip(-e, 0, None)   # 少报量
    over = np.clip(e, 0, None)     # 多报量
    day = rows["ghi"].values > 0   # 只统计有日照的断面（夜间两边都是 0，没有方向问题）

    def m(x, mask):
        return float(x[mask].mean()) if mask.any() else np.nan

    return dict(
        station=sid, 设定=tag,
        少报比例=round((e < -1e-9).sum() / n * 100, 2),
        多报比例=round((e > 1e-9).sum() / n * 100, 2),
        完全准确比例=round((np.abs(e) <= 1e-9).sum() / n * 100, 2),
        平均带符号误差bias=round(m(e, day), 4),
        少报均值_白天=round(m(under, day), 4),
        多报均值_白天=round(m(over, day), 4),
        少报累计_白天=round(m(under, day) * day.sum(), 4),
        多报累计_白天=round(m(over, day) * day.sum(), 4),
        少报峰值_白天=round(m(under, day), 4),
        n断面=n, n白天=int(day.sum()),
    )


def run_setting(d, feats, tag, baseline_fn, sid):
    cols = feats + [c for c in ["p_norm", "date_time", GHI, NWP_GHI] if c not in feats]
    dd = d[cols].dropna().reset_index(drop=True)
    tr, te = split_by_time(dd)
    out = []

    def pack(name, pred):
        rows = pd.DataFrame({"true": te["p_norm"].values, "pred": pred,
                             "ghi": te[GHI].values if GHI in te.columns
                             else te[NWP_GHI].values if NWP_GHI in te.columns else 0.0,
                             "hour": te["date_time"].dt.hour.values})
        r = direction_rows(sid, tag, rows)
        r["方法"] = name
        out.append(r)

    name, pred = baseline_fn(tr, te)
    if isinstance(pred, tuple):
        pred = pred[0]
    pack(name, np.asarray(pred, dtype=float))

    Xtr, ytr = tr[feats].values, tr["p_norm"].values
    Xte = te[feats].values
    for name, mdl in make_models().items():
        mdl.fit(Xtr, ytr)
        pack(name, np.asarray(mdl.predict(Xte), dtype=float))
    return out


def main():
    all_rows = []
    for sid in STATIONS:
        d = load_station(sid)
        cap = capacity_mw(sid)
        print(f"[{sid}] {len(d)} 条，{cap:.1f} MW")

        all_rows += run_setting(add_time_features(d.copy()), feats_a_lmd(),
                                "甲-同时刻实测(lmd)", bl_a_lmd, sid)
        all_rows += run_setting(add_time_features(d.copy()), feats_a_nwp(),
                                "甲-同时刻预报(nwp)", bl_a_nwp, sid)
        db, fb = build_b(d)
        all_rows += run_setting(db, fb, "乙-纯历史时序", bl_b, sid)
        dc, fc = build_c(d)
        all_rows += run_setting(dc, fc, "丙-同时刻气象+历史", bl_b, sid)

    per = pd.DataFrame(all_rows)
    per.to_csv(os.path.join(RES, "pvod_error_direction.csv"), index=False, encoding="utf-8-sig")

    # ---------- 十站平均 ----------
    def agg(g):
        n_day = g["n白天"].mean()
        return pd.Series({
            "少报比例": round(g["少报比例"].mean(), 2),
            "多报比例": round(g["多报比例"].mean(), 2),
            "bias(白天)": round(g["平均带符号误差bias"].mean(), 4),
            "少报均值(白天)": round(g["少报均值_白天"].mean(), 4),
            "多报均值(白天)": round(g["多报均值_白天"].mean(), 4),
            "少报累计(白天)": round(g["少报累计_白天"].mean(), 2),
            "多报累计(白天)": round(g["多报累计_白天"].mean(), 2),
            "n白天": int(n_day),
        })
    avg = per.groupby(["设定", "方法"]).apply(agg, include_groups=False).reset_index()
    avg.to_csv(os.path.join(RES, "pvod_error_direction_summary.csv"), index=False, encoding="utf-8-sig")

    # ---------- 折算成钱 ----------
    total_mw = sum(capacity_mw(s) for s in STATIONS)
    avg_mw = total_mw / len(STATIONS)             # 与「十站平均」口径配套，用单站平均容量
    hours = per["n断面"].mean() * 0.25            # 单站测试期小时数
    days = hours / 24.0
    annual_factor = 365.0 / max(days, 1e-9)       # 外推到满一年

    cost_rows = []
    for _, r in avg.iterrows():
        u = r["少报累计(白天)"]      # 归一化量（%装机容量 → 0~1）
        o = r["多报累计(白天)"]
        # 归一化量 × 单站平均装机(MW) × 1000(kW/MW) × 0.25h = kWh
        kwh_under = u * avg_mw * 1000 * 0.25
        kwh_over = o * avg_mw * 1000 * 0.25
        cost_rows.append(dict(
            设定=r["设定"], 方法=r["方法"],
            bias=round(r["bias(白天)"], 4),
            少报kWh_测试期=round(kwh_under, 1),
            多报kWh_测试期=round(kwh_over, 1),
            少报代价元_测试期=round(kwh_under * P_PV, 1),
            多报代价元_测试期=round(kwh_over * P_COAL, 1),
            多报代价元_现货上限=round(kwh_over * P_PEAK, 1),
            少报代价元_年化=round(kwh_under * P_PV * annual_factor, 0),
            多报代价元_年化=round(kwh_over * P_COAL * annual_factor, 0),
            多报代价元_年化_现货上限=round(kwh_over * P_PEAK * annual_factor, 0),
        ))
    cost = pd.DataFrame(cost_rows)
    cost["多报/少报 代价比"] = (cost["多报代价元_测试期"] / cost["少报代价元_测试期"].replace(0, np.nan)).round(2)
    cost.to_csv(os.path.join(RES, "pvod_error_direction_cost.csv"), index=False, encoding="utf-8-sig")

    meta = dict(total_mw=total_mw, avg_station_mw=round(avg_mw, 3),
                test_hours=round(hours, 1), test_days=round(days, 1),
                annual_factor=round(annual_factor, 2), prices=PRICES)
    pd.Series(meta).to_csv(os.path.join(RES, "pvod_error_cost_meta.csv"),
                           header=["值"], encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("【十站平均 · 误差分方向】单位：%装机容量；少报=预测偏低，多报=预测偏高")
    print(avg.to_string(index=False))
    print("\n【折算成钱】")
    print(cost.to_string(index=False))
    print(f"\n十站总装机 {total_mw} MW；单站测试期 {hours:.0f} 小时 ≈ {days:.0f} 天；"
          f"年化系数 {annual_factor:.2f}")
    print("\n已保存：results/pvod_error_direction*.csv")


if __name__ == "__main__":
    main()
