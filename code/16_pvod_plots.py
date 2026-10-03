# -*- coding: utf-8 -*-
"""
16_pvod_plots.py
================
把 PVOD 的全部结果画成报告用图，覆盖 fig1~fig7，另加一张时区证据图 fig10。

图清单：
  fig1  典型日出力曲线（晴 / 多云 / 阴 各一天，功率与辐射双轴）
  fig2  全年逐日峰值出力（归一化，含 7 日滑动平均，看季节变化）
  fig3  辐射-功率散点图（题目明确要求的一张）
  fig4  气象因子-出力 相关系数热力图
  fig5  天气分场景下的模型增益
  fig6  特征重要性（三个设定并排）
  fig7  特征消融（含设定丙「删掉同时刻气象」这一关键对照）
  fig10 时区证据：同一批数据按 UTC 与按北京时间统计的日变化曲线
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
from pvod_common import (BASE, RES, FIG, STATIONS, GHI, load_station,
                         add_time_features, LMD_FEATS, NWP_FEATS)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

os.makedirs(FIG, exist_ok=True)
META = pd.read_csv(os.path.join(BASE, "data", "PVOD_hebei", "metadata.csv")).set_index("Station_ID")


def solar_geometry(dt_index, lat, lon):
    """大气层外水平面辐射 I0h 与太阳高度角正弦（输入为北京时间）。"""
    n = dt_index.dt.dayofyear.values.astype(float)
    delta = np.deg2rad(23.45 * np.sin(np.deg2rad(360 * (284 + n) / 365)))
    phi = np.deg2rad(lat)
    hour_dec = dt_index.dt.hour.values + dt_index.dt.minute.values / 60.0
    omega = np.deg2rad(15.0 * (hour_dec + (lon - 120.0) / 15.0 - 12.0))
    sin_h = np.clip(np.sin(phi) * np.sin(delta)
                    + np.cos(phi) * np.cos(delta) * np.cos(omega), 0.0, None)
    I0 = 1367.0 * (1 + 0.033 * np.cos(np.deg2rad(360 * n / 365)))
    return I0 * sin_h, sin_h


# ----------------------------------------------------------------- fig1
def flag_flat_days(d, min_run=8, min_val=100.0):
    """标出「平顶日」——原始总辐射连续多个点完全相等（缺测被常数填充的痕迹）。
    station01 的 2019-06-10 就是典型：10:00~19:00 总辐射恒为 837.80 W/m²。
    选典型日时必须避开这类日子，否则画出来是一条方波，会让人误以为模型有问题。"""
    v = d[GHI].values
    new = np.r_[True, v[1:] != v[:-1]]
    grp = np.cumsum(new)
    run = np.bincount(grp)[grp]
    d["flat"] = (run >= min_run) & (v > min_val)
    frac = d.groupby(d["date_time"].dt.date)["flat"].mean()
    return set(frac[frac > 0.2].index)


def fig1_typical_day(sid="station01"):
    d = load_station(sid)
    lat, lon = META.loc[sid, "Latitude"], META.loc[sid, "Longitude"]
    I0h, _ = solar_geometry(d["date_time"], lat, lon)
    d["kt"] = np.where(I0h > 50, d[GHI] / np.where(I0h > 0, I0h, np.nan), np.nan)

    flat_days = flag_flat_days(d)
    day = d.groupby(d["date_time"].dt.date).agg(kt=("kt", "mean"),
                                                pk=("p_norm", "max"))
    day = day[day["kt"].notna() & (day["pk"] > 0.05)]
    day = day[~day.index.isin(flat_days)]
    sunny = day["kt"].idxmax()
    cloudy = day["kt"].idxmin()
    mid = (day["kt"] - 0.55).abs().idxmin()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, (dd, tag) in zip(axes, [(sunny, "晴天"), (mid, "多云"), (cloudy, "阴天")]):
        s = d[d["date_time"].dt.date == dd]
        t = s["date_time"].dt.hour + s["date_time"].dt.minute / 60.0
        ax.plot(t, s["p_norm"] * 100, color="#c0392b", lw=1.8, label="出力 (%容量)")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlabel("北京时间 (h)")
        ax.grid(alpha=0.25, ls=":")
        ax2 = ax.twinx()
        ax2.plot(t, s[GHI], color="#2980b9", lw=1.4, ls="--", label="总辐射")
        ax2.set_ylabel("总辐射 (W/m²)")
        ax.set_title(f"{tag}    {dd}\n日峰值出力 {s['p_norm'].max()*100:.1f}% 容量",
                     fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("出力 (%装机容量)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.suptitle(f"图1  {sid}（{META.loc[sid,'Capacity']/1000:.0f} MW）三类典型日的出力与辐射",
                 fontsize=11, y=1.04)
    fig.savefig(os.path.join(FIG, "fig1_typical_day.png"))
    plt.close(fig)
    print(f"fig1 典型日曲线 OK（已排除平顶异常日 {sorted(str(x) for x in flat_days)}）")


# ----------------------------------------------------------------- fig2
def fig2_daily_max(sid="station01"):
    d = load_station(sid)
    day = d.groupby(d["date_time"].dt.date)["p_norm"].max()
    day.index = pd.to_datetime(day.index)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(day.index, day.values * 100, color="#bdc3c7", lw=0.8, label="逐日峰值")
    ax.plot(day.index, (day.rolling(7, min_periods=3).mean() * 100),
            color="#c0392b", lw=1.8, label="7 日滑动平均")
    ax.set_ylabel("日峰值出力 (%装机容量)")
    ax.set_xlabel("日期")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9)
    ax.set_title(f"图2  {sid} 全年逐日峰值出力（{day.index.min().date()} ~ {day.index.max().date()}）",
                 fontsize=11)
    fig.savefig(os.path.join(FIG, "fig2_daily_max.png"))
    plt.close(fig)
    print("fig2 全年逐日峰值 OK")


# ----------------------------------------------------------------- fig3
def fig3_scatter():
    """题目明确要求的一张：辐射-功率散点。十站合并后抽样，避免点太密。"""
    rng = np.random.default_rng(42)
    xs, ys = [], []
    for sid in STATIONS:
        d = load_station(sid)
        s = d[d[GHI] > 20]
        if len(s) == 0:
            continue
        idx = rng.choice(len(s), size=min(2500, len(s)), replace=False)
        ss = s.iloc[idx]
        xs.append(ss[GHI].values)
        ys.append(ss["p_norm"].values * 100)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    k = float(np.sum(x * y) / np.sum(x ** 2))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(x, y, s=3, alpha=0.12, color="#2980b9", edgecolors="none")
    xx = np.linspace(0, x.max(), 100)
    ax.plot(xx, k * xx, color="#c0392b", lw=2, label=f"固定系数拟合  P≈{k:.4f}·GHI")
    ax.set_xlabel("总辐射 GHI (W/m²)")
    ax.set_ylabel("出力 (%装机容量)")
    ax.set_title("图3  十站合并的辐射-出力散点（每站抽样 2500 点）", fontsize=11)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9)
    r = float(np.corrcoef(x, y)[0, 1])
    ax.text(0.05, 0.92, f"相关系数 r = {r:.4f}\n散点明显有宽度，并非一条直线",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(fc="#fff8e1", ec="#cccccc"))
    fig.savefig(os.path.join(FIG, "fig3_scatter.png"))
    plt.close(fig)
    print(f"fig3 散点图 OK（r={r:.4f}，拟合系数 k={k:.5f}）")


# ----------------------------------------------------------------- fig4
def fig4_corr():
    cols = LMD_FEATS + NWP_FEATS
    name = {"lmd_totalirrad": "实测总辐射", "lmd_diffuseirrad": "实测散射",
            "lmd_temperature": "实测气温", "lmd_pressure": "实测气压",
            "lmd_windspeed": "实测风速", "lmd_winddirection": "实测风向",
            "nwp_globalirrad": "预报总辐射", "nwp_directirrad": "预报直接辐射",
            "nwp_temperature": "预报气温", "nwp_humidity": "预报湿度",
            "nwp_windspeed": "预报风速", "nwp_winddirection": "预报风向",
            "nwp_pressure": "预报气压", "power": "出力"}
    mats = []
    for sid in STATIONS:
        d = load_station(sid)
        mats.append(d[cols + ["power"]].corr())
    C = sum(mats) / len(mats)
    C.index = [name.get(i, i) for i in C.index]
    C.columns = [name.get(i, i) for i in C.columns]

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C)))
    ax.set_xticklabels(C.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(C)))
    ax.set_yticklabels(C.index, fontsize=8.5)
    for i in range(len(C)):
        for j in range(len(C)):
            v = C.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("图4  气象因子与出力的相关系数（十站平均）", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig4_corr.png"))
    plt.close(fig)
    print("fig4 相关系数热力图 OK")


# ----------------------------------------------------------------- fig5
def fig5_weather():
    w = pd.read_csv(os.path.join(RES, "pvod_weather_scenarios.csv"))
    w = w[w["口径"] == "有效发电(辐射>50)"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, dim in zip(axes, ["晴度", "辐射突变", "光照强度"]):
        sub = w[w["维度"] == dim].copy()
        order = {"晴度": ["阴 (kt<0.35)", "多云 (0.35≤kt<0.6)", "晴 (kt≥0.6)"],
                 "辐射突变": ["平稳 <10", "缓变 10~30", "突变 30~60", "剧变 >60"],
                 "光照强度": ["弱光 <200", "中光 200~600", "强光 ≥600"]}[dim]
        sub["分组"] = pd.Categorical(sub["分组"], order, ordered=True)
        sub = sub.sort_values("分组")
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["基线_MAE"], width=0.4, color="#95a5a6", label="Persistence 基线")
        ax.bar(x + 0.2, sub["RF_MAE"], width=0.4, color="#2980b9", label="随机森林")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\nn={n}" for g, n in zip(sub["分组"], sub["样本数"])],
                           fontsize=8)
        ax.set_ylabel("MAE (%装机容量)")
        ax.grid(alpha=0.25, ls=":", axis="y")
        ax.set_title(dim, fontsize=10.5)
        for xi, (rf, bs) in zip(x, zip(sub["RF_MAE"], sub["基线_MAE"])):
            dlt = (bs - rf) / bs * 100
            ax.text(xi + 0.2, rf + 0.15, f"{dlt:+.0f}%", ha="center",
                    fontsize=8, color="#c0392b" if dlt > 0 else "#27ae60")
        if ax is axes[0]:
            ax.legend(fontsize=8.5)
    fig.suptitle("图5  天气分场景下随机森林相对基线的误差（十站测试集，有效发电时段）",
                 fontsize=11)
    fig.savefig(os.path.join(FIG, "fig5_weather_scenarios.png"))
    plt.close(fig)
    print("fig5 天气分场景 OK")


# ----------------------------------------------------------------- fig6
def fig6_importance():
    imp = pd.read_csv(os.path.join(RES, "pvod_importance.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, tag in zip(axes, ["丙-同时刻气象+历史", "乙-纯历史时序", "甲-同时刻实测(lmd)"]):
        sub = imp[imp["设定"] == tag].sort_values("重要性")
        ax.barh(sub["中文名"], sub["重要性"], color="#2980b9",
                xerr=sub["站间标准差"], error_kw=dict(ecolor="#7f8c8d", lw=0.9, capsize=2))
        ax.set_xlabel("随机森林特征重要性")
        ax.set_title(tag, fontsize=10)
        ax.grid(alpha=0.25, ls=":", axis="x")
        ax.tick_params(labelsize=8.5)
    fig.suptitle("图6  特征重要性（十站平均，误差棒为站间标准差）", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig6_feature_importance.png"))
    plt.close(fig)
    print("fig6 特征重要性 OK")


# ----------------------------------------------------------------- fig7
def fig7_ablation():
    ab = pd.read_csv(os.path.join(RES, "pvod_ablation.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)
    for ax, tag in zip(axes, ["丙-同时刻气象+历史", "乙-纯历史时序", "甲-同时刻实测(lmd)"]):
        sub = ab[ab["设定"] == tag].copy()
        sub = sub.sort_values("MAE")
        colors = ["#27ae60" if v <= 0.001 else "#c0392b" for v in sub["相对完整变化"]]
        ax.barh(sub["方案"], sub["MAE"], color=colors)
        for i, (m, dl) in enumerate(zip(sub["MAE"], sub["相对完整变化"])):
            ax.text(m + 0.03, i, f"{m:.3f}" + ("" if abs(dl) < 0.001 else f"  ({dl:+.2f}%)"),
                    va="center", fontsize=8)
        ax.set_xlabel("MAE (%装机容量)")
        ax.set_title(tag, fontsize=10)
        ax.grid(alpha=0.25, ls=":", axis="x")
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, sub["MAE"].max() * 1.45)
    fig.suptitle("图7  特征消融：整组删掉某类特征后的误差（绿色=完整方案，红色=删减后）",
                 fontsize=11)
    fig.savefig(os.path.join(FIG, "fig7_ablation.png"))
    plt.close(fig)
    print("fig7 消融 OK")


# ----------------------------------------------------------------- fig10
def fig10_timezone(sid="station01"):
    """时区证据：同一份原始时间戳，直接按时钟小时统计 vs 换算成北京时间后统计。"""
    raw = pd.read_csv(os.path.join(BASE, "data", "PVOD_hebei", f"{sid}.csv"))
    raw["date_time"] = pd.to_datetime(raw["date_time"])
    g_utc = raw.groupby(raw["date_time"].dt.hour)[GHI].mean()
    g_cst = raw.groupby((raw["date_time"] + pd.Timedelta(hours=8)).dt.hour)[GHI].mean()

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.plot(g_utc.index, g_utc.values, "o-", color="#7f8c8d", lw=1.6,
            label="按原始时间戳统计（实为 UTC）")
    ax.plot(g_cst.index, g_cst.values, "s-", color="#c0392b", lw=1.8,
            label="换算为北京时间（+8 h）")
    ax.set_xlabel("时钟小时 (h)")
    ax.set_ylabel("平均总辐射 (W/m²)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9)
    ax.set_title(f"图10  {sid} 总辐射的日变化：原始时间戳的峰值出现在 "
                 f"{g_utc.idxmax():d} 时，换算后落在 {g_cst.idxmax():d} 时", fontsize=10.5)
    fig.savefig(os.path.join(FIG, "fig10_timezone.png"))
    plt.close(fig)
    print(f"fig10 时区证据 OK（UTC 峰值 {g_utc.idxmax()} 时 → 北京时间 {g_cst.idxmax()} 时）")


if __name__ == "__main__":
    fig1_typical_day()
    fig2_daily_max()
    fig3_scatter()
    fig4_corr()
    fig5_weather()
    fig6_importance()
    fig7_ablation()
    fig10_timezone()
    print("\n全部图片已输出到 figures/")
