# -*- coding: utf-8 -*-
"""
PVOD 河北十站 —— 皮尔逊相关性分析（任务清单次要第 1 项）

做三件事：
  1. 逐站算皮尔逊相关矩阵，画成 10 张小热力图（fig11），看十个站的相关结构是否一致；
  2. 算每个变量与出力的相关系数在十站上的均值与标准差（fig12），
     误差棒短 = 该变量的作用在各站稳定，误差棒长 = 依赖站点自身条件；
  3. 算数值天气预报 nwp_* 与站内实测 lmd_* 同名变量的相关与偏差（fig13），
     用来解释「为什么把实测气象换成预报气象后误差会翻倍」。

【为什么要单独做这一步】
  相关矩阵不等于因果，也不等于模型里的特征重要性。
  它的用处是：先看清楚变量两两之间"长得像不像"（共线程度），
  再回头解释为什么删掉气温类特征模型几乎不变——
  因为气温和辐射高度相关，模型分不开二者。
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

from pvod_common import (BASE, RES, FIG, STATIONS, GHI, load_station, NAME_CN)

plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# 逐站小热力图用到的核心变量（用归一化出力 p_norm，跨站可比）
CORE = ["lmd_totalirrad", "lmd_diffuseirrad", "lmd_temperature",
        "lmd_pressure", "lmd_windspeed", "p_norm"]
CORE_CN = ["总辐射", "散射辐射", "气温", "气压", "风速", "出力"]

# 预报 vs 实测 的配对
PAIRS = [("nwp_globalirrad", "lmd_totalirrad", "总辐射"),
         ("nwp_temperature", "lmd_temperature", "气温"),
         ("nwp_pressure", "lmd_pressure", "气压"),
         ("nwp_windspeed", "lmd_windspeed", "风速")]


def main():
    os.makedirs(FIG, exist_ok=True)

    # ---------------- 1. 逐站相关矩阵 ----------------
    mats = {}
    for sid in STATIONS:
        d = load_station(sid)
        mats[sid] = d[CORE].corr(method="pearson")

    fig, axes = plt.subplots(2, 5, figsize=(16.5, 7.0))
    for k, (ax, sid) in enumerate(zip(axes.ravel(), STATIONS)):
        C = mats[sid].values
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(CORE_CN)))
        ax.set_xticklabels(CORE_CN, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(CORE_CN)))
        # 只有第一列标 y 轴变量名，避免十张图重复堆叠
        ax.set_yticklabels(CORE_CN if k % 5 == 0 else [], fontsize=8)
        for i in range(len(CORE)):
            for j in range(len(CORE)):
                v = C[i, j]
                if i == j:
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                        color="white" if abs(v) > 0.55 else "black")
        cap = d_cap(sid)
        ax.set_title(f"{sid}（{cap:.0f} MW）", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    fig.suptitle("图11  十站各自的皮尔逊相关系数矩阵（对角线与自身相关恒为 1，已省略）",
                 fontsize=12)
    fig.savefig(os.path.join(FIG, "fig11_pearson_by_station.png"))
    plt.close(fig)
    print("fig11 逐站相关矩阵 OK")

    # ---------------- 2. 与出力的相关：十站均值 ± 标准差 ----------------
    rows = []
    for v, cn in zip(CORE[:-1], CORE_CN[:-1]):
        vals = [mats[sid].loc[v, "p_norm"] for sid in STATIONS]
        rows.append(dict(变量=cn, 均值=float(np.mean(vals)),
                         标准差=float(np.std(vals, ddof=1)),
                         最小=float(np.min(vals)), 最大=float(np.max(vals))))
    stat = pd.DataFrame(rows).sort_values("均值", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ypos = np.arange(len(stat))[::-1]
    ax.barh(ypos, stat["均值"], xerr=stat["标准差"], align="center",
            color="#2980b9", alpha=0.85, error_kw=dict(ecolor="#c0392b", capsize=4, lw=1.3))
    ax.set_yticks(ypos)
    ax.set_yticklabels(stat["变量"], fontsize=10)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("与归一化出力的皮尔逊相关系数（十站均值，误差棒为站间标准差）")
    ax.grid(axis="x", alpha=0.3, ls=":")
    for y, m, s in zip(ypos, stat["均值"], stat["标准差"]):
        ax.text(m + (s + 0.02 if m >= 0 else -s - 0.02), y, f"{m:+.3f}",
                va="center", ha="left" if m >= 0 else "right", fontsize=9)
    ax.set_title("图12  各气象因子与出力的相关强度（十站汇总）", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig12_corr_stability.png"))
    plt.close(fig)
    print("fig12 相关强度汇总 OK")

    # ---------------- 3. 预报 vs 实测 ----------------
    prows = []
    for sid in STATIONS:
        d = load_station(sid)
        for nw, lm, cn in PAIRS:
            if nw not in d.columns or lm not in d.columns:
                continue
            a, b = d[nw].values, d[lm].values
            ok = np.isfinite(a) & np.isfinite(b)
            r = float(np.corrcoef(a[ok], b[ok])[0, 1])
            rmse = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
            # 归一化 RMSE：除以实测量的标准差
            nrmse = rmse / float(np.std(b[ok])) if np.std(b[ok]) > 0 else np.nan
            prows.append(dict(station=sid, 变量=cn, 相关=r, RMSE=rmse, 归一化RMSE=nrmse))
    P = pd.DataFrame(prows)
    P.to_csv(os.path.join(RES, "pvod_nwp_vs_lmd.csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    piv_r = P.pivot(index="station", columns="变量", values="相关")
    piv_n = P.pivot(index="station", columns="变量", values="归一化RMSE")
    for ax, piv, ttl in [(axes[0], piv_r, "预报与实测的相关系数"),
                         (axes[1], piv_n, "预报误差 nRMSE（除以实测标准差）")]:
        piv.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel(ttl)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3, ls=":")
        ax.legend(fontsize=8)
    fig.suptitle("图13  数值天气预报 nwp_* 与站内实测 lmd_* 的一致程度", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig13_nwp_vs_lmd.png"))
    plt.close(fig)
    print("fig13 预报vs实测 OK")

    # ---------------- 3.5 偏相关：控制总辐射后还剩多少 ----------------
    # 气温与出力呈正相关（+0.35），但物理上组件升温会降低效率，二者矛盾。
    # 合理的解释是「夏天既热又晒」造成的共线伪相关。
    # 一阶偏相关 r(xy|z) = (r_xy - r_xz·r_yz) / sqrt((1-r_xz²)(1-r_yz²))
    prows2 = []
    for sid in STATIONS:
        C = mats[sid]
        for v, cn in zip(CORE[:-1], CORE_CN[:-1]):
            if v == "lmd_totalirrad":
                continue
            r_xy = C.loc[v, "p_norm"]
            r_xz = C.loc[v, "lmd_totalirrad"]
            r_yz = C.loc["lmd_totalirrad", "p_norm"]
            den = np.sqrt(max((1 - r_xz ** 2) * (1 - r_yz ** 2), 1e-12))
            pr = (r_xy - r_xz * r_yz) / den
            prows2.append(dict(station=sid, 变量=cn,
                               边缘相关=r_xy, 偏相关=pr))
    PC = pd.DataFrame(prows2)
    ps = (PC.groupby("变量", as_index=False)
            .agg(边缘相关=("边缘相关", "mean"), 偏相关=("偏相关", "mean"),
                 偏相关标准差=("偏相关", lambda s: float(np.std(s, ddof=1)))))
    ps["变化"] = ps["偏相关"] - ps["边缘相关"]
    ps = ps.sort_values("偏相关", key=abs, ascending=False)
    ps.to_csv(os.path.join(RES, "pvod_partial_corr.csv"),
              index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    xv = np.arange(len(ps))
    w = 0.38
    ax.bar(xv - w / 2, ps["边缘相关"], w, label="边缘相关（未控制）", color="#7f8c8d", alpha=0.9)
    ax.bar(xv + w / 2, ps["偏相关"], w, yerr=ps["偏相关标准差"], capsize=3,
           label="偏相关（控制总辐射后）", color="#c0392b", alpha=0.9,
           error_kw=dict(ecolor="#2c3e50", lw=1.1))
    ax.set_xticks(xv)
    ax.set_xticklabels(ps["变量"], fontsize=10)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("相关系数")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.set_title("图14  控制总辐射后各因子与出力的偏相关（十站均值）", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig14_partial_corr.png"))
    plt.close(fig)
    print("fig14 偏相关 OK")

    print("\n偏相关（控制总辐射后，十站均值）")
    print(ps.round(4).to_string(index=False))

    # ---------------- 结果表 ----------------
    out = []
    for sid in STATIONS:
        C = mats[sid]
        for v, cn in zip(CORE[:-1], CORE_CN[:-1]):
            out.append(dict(station=sid, 变量=cn, 与出力相关=round(C.loc[v, "p_norm"], 4),
                            与总辐射相关=round(C.loc[v, "lmd_totalirrad"], 4)
                            if v != "lmd_totalirrad" else 1.0))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(RES, "pvod_pearson.csv"), index=False, encoding="utf-8-sig")
    stat.to_csv(os.path.join(RES, "pvod_pearson_summary.csv"),
                index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("各气象因子与出力的皮尔逊相关（十站均值 ± 标准差，按绝对值排序）")
    print(stat.round(4).to_string(index=False))
    print("\n预报 vs 实测（十站平均）")
    print(P.groupby("变量")[["相关", "归一化RMSE"]].mean().round(4).to_string())
    print("\n已保存：results/pvod_pearson.csv / pvod_pearson_summary.csv / pvod_nwp_vs_lmd.csv")


def d_cap(sid):
    """从 metadata.csv 取该站装机容量并换算成 MW。
    metadata 的列名是 Station_ID / Capacity（容量单位 kW）。"""
    try:
        meta = pd.read_csv(os.path.join(BASE, "data", "PVOD_hebei", "metadata.csv"),
                           encoding="utf-8-sig")
        col = "Station_ID" if "Station_ID" in meta.columns else meta.columns[0]
        row = meta[meta[col] == sid]
        if len(row) == 0:
            row = meta.iloc[int(sid[-2:])]
        return float(row.iloc[0]["Capacity"]) / 1000.0
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
