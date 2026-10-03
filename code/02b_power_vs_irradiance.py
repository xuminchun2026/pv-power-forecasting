# -*- coding: utf-8 -*-
# 2026-10-03：本报告最终改用 PVOD 数据集，本脚本为早期用另一份候选数据集做的探索，已弃用；
# 正式结果见 10_pvod_load_check.py 起的各脚本。
"""
第 2.5 步：目标列「实际功率」到底是不是实测值？—— 数据体检（forensic）
=========================================================================
数据集：data/ningxia_2019_15min.xlsx
目标列：实际功率（kW）

背景：画图时发现「功率」和「总辐射 GHI」几乎贴成一条直线。
      真实电站的出力不可能这么规整（温度、云团、逆变器、灰尘都会扰动）。
      本脚本用 4 组独立检验来判断：这一列是**电表实测**，还是
      **按 P = 辐射 × 系数 换算出来的理论值**。

怎么运行：PyCharm 里右键本文件 -> 运行 '02b_power_vs_irradiance'
=========================================================================
"""
import pandas as pd
import numpy as np

DATA_PATH = "/Users/xuminchun/Desktop/电气club作业/data/ningxia_2019_15min.xlsx"
TIME_COL = "采样结束时刻"
GHI_COL  = "GHI（太阳总水平辐射）"      # 水平面上的太阳总辐射
TARGET   = "实际功率"                   # 要研究的目标列

df = pd.read_excel(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])

print("=" * 60)
print("检验 1：同一个辐射值，功率是不是只对应同一个数？")
print("  真实电站：同一辐射下，不同时刻的功率必然有高有低（云、温度、灰尘）")
print("  理论换算：同一辐射只会对应一个固定的功率值\n")
for g in [200, 300, 400, 500, 600, 700, 800, 900]:
    sub = df[df[GHI_COL] == g]
    if len(sub) == 0:
        continue
    u = np.sort(sub[TARGET].unique())
    print("  GHI=%4d : 出现 %3d 次 -> 功率只取 %2d 个值  "
          "[%.4f ~ %.4f kW]  散开 %.2f%%"
          % (g, len(sub), len(u), u.min(), u.max(),
             100 * (u.max() - u.min()) / u.mean()))

print("\n" + "=" * 60)
print("检验 2：功率/辐射 这个比值，会不会随天气变化？")
d = df[df[GHI_COL] > 50].copy()
d["ratio"] = d[TARGET] / d[GHI_COL]
r = d["ratio"]
print("  比值均值 = %.6f  标准差 = %.6f  变异系数 = %.4f%%"
      % (r.mean(), r.std(), 100 * r.std() / r.mean()))
print("  （实测数据的这个比值通常波动 10%~30%；这是 0.2% 量级）")

d["晴空指数"] = d["DNI（太阳直接辐射指数）"] / d[GHI_COL]   # 越大越晴
d["box"] = pd.cut(d["晴空指数"], [0, .3, .5, .7, .85, 1.01],
                  labels=["很阴", "偏阴", "半晴", "晴", "很晴"])
print("\n  按晴天/阴天分组看比值（真实电站阴天出力会明显打折）：")
print(d.groupby("box", observed=True)["ratio"].mean().round(6).to_string())

print("\n" + "=" * 60)
print("检验 3：温度效应在不在？（光伏板温度每升 1°C，效率约降 0.4%）")
d3 = df[(df[GHI_COL] > 400) & (df[GHI_COL] < 600)].copy()  # 只比辐射相近的，排除混淆
d3["ratio"] = d3[TARGET] / d3[GHI_COL]
d3["温度档"] = pd.cut(d3["气温"], [-20, 5, 15, 25, 35, 60],
                      labels=["低于5度", "5到15度", "15到25度", "25到35度", "高于35度"])
t = d3.groupby("温度档", observed=True)["ratio"].mean()
print(t.round(6).to_string())
print("  最冷档 vs 最热档 实际差异 = %.2f%%"
      % (100 * (t.iloc[-1] - t.iloc[0]) / t.iloc[0]))
print("  按 -0.4%%/度 计算，两档（差约 35 度）本应差约 %.0f%%" % (0.4 * 35))

print("\n" + "=" * 60)
print("检验 4：用最小二乘看整条关系")
G = df[GHI_COL].values.astype(float)
P = df[TARGET].values.astype(float)
k, b = np.polyfit(G, P, 1)
res = P - (k * G + b)
print("  P ≈ %.6f × GHI %+ .5f" % (k, b))
print("  残差最大 = %.4f kW，占满发(%.1f kW) 的 %.2f%%"
      % (np.abs(res).max(), P.max(), 100 * np.abs(res).max() / P.max()))
for c in ["气温", "云层不透明度", "高度10m风速"]:
    print("  残差 与 %-8s 相关系数 = %+.4f" % (c, np.corrcoef(res, df[c].values)[0, 1]))

print("\n" + "=" * 60)
print("[结论] 四项检验一致指向：实际功率 ≈ 固定系数 × GHI，")
print("       与天气、温度几乎无关 -> 该列很可能是按公式换算的参考功率，")
print("       而不是电表实测值。此结论需向数据提供方（学长/负责人）核实后再写进报告。")
print("=" * 60)
