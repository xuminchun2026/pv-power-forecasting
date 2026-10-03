# -*- coding: utf-8 -*-
"""
15_pvod_target_check.py
=======================
目标列性质检验：PVOD 的 power 究竟是真实计量的电表读数，还是由辐射换算出来的？

在宁夏那份数据上我们曾发现 P ≈ 0.1255 × GHI、比值变异系数仅 0.23%，
几乎是一个确定性公式，因此怀疑那列是换算值而非实测值。
本脚本把同一套检验搬到 PVOD 上：

  ① 拟合 P_norm = k × GHI（无截距），看残差有多大
  ② 逐条算比值 ratio = power / GHI，看它的变异系数（CV）有多大
  ③ 看 ratio 与气温的相关系数
     —— 真实光伏有温度效应（约 -0.3~-0.4 %/℃），ratio 应随气温系统性下降

若 PVOD 的比值 CV 远大于宁夏、且残留温度痕迹，说明这里才是真实计量值。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvod_common import (RES, STATIONS, GHI, load_station, fit_coef)

MIN_GHI = 50.0   # 日出日落附近辐射太小，比值不稳定，只看白天的有效样本


def main():
    rows = []
    for sid in STATIONS:
        d = load_station(sid)
        day = d[d[GHI] > MIN_GHI].copy()
        if len(day) < 100:
            continue

        k = fit_coef(day, GHI)
        pred = k * day[GHI].values
        resid = day["p_norm"].values - pred

        ratio = day["power"].values / day[GHI].values        # MW per W/m²
        cv = float(np.std(ratio) / np.mean(ratio) * 100)

        corr_t = float(np.corrcoef(ratio, day["lmd_temperature"].values)[0, 1])
        corr_ghi = float(np.corrcoef(day["power"].values, day[GHI].values)[0, 1])

        # 控制辐射后的偏相关：先把 ratio 里能被辐射解释的部分扣掉，
        # 再看残差与气温的关系。否则气温与辐射本身高度相关，看不出独立作用。
        g = day[GHI].values
        A = np.column_stack([np.ones_like(g), g, g ** 2])
        coef, *_ = np.linalg.lstsq(A, ratio, rcond=None)
        ratio_resid = ratio - A @ coef
        temp = day["lmd_temperature"].values
        partial_t = float(np.corrcoef(ratio_resid, temp)[0, 1])
        # 温度效应斜率：ratio 残差对气温做一元回归，换算成「每升高 1 ℃ 比值变化百分之几」
        slope = float(np.polyfit(temp, ratio_resid, 1)[0])
        temp_effect = slope / float(np.mean(ratio)) * 100

        rows.append(dict(
            station=sid,
            容量MW=round(d.attrs["cap_mw"], 1),
            白天样本=len(day),
            系数k=round(k, 8),
            残差最大=round(float(np.max(np.abs(resid))) * 100, 4),
            残差RMSE=round(float(np.sqrt(np.mean(resid ** 2))) * 100, 4),
            比值CV百分比=round(cv, 3),
            比值_气温相关=round(corr_t, 4),
            控辐射后_气温相关=round(partial_t, 4),
            温度效应_每度百分比=round(temp_effect, 4),
            功率_辐射相关=round(corr_ghi, 4),
        ))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "pvod_target_check.csv"), index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("目标列性质检验（PVOD 十站，只取总辐射 > 50 W/m² 的白天样本）")
    print(out.to_string(index=False))
    print()
    print(f"比值变异系数 CV：{out['比值CV百分比'].min():.2f}% ~ {out['比值CV百分比'].max():.2f}%，"
          f"十站平均 {out['比值CV百分比'].mean():.2f}%")
    print(f"比值与气温相关系数：{out['比值_气温相关'].min():.4f} ~ {out['比值_气温相关'].max():.4f}，"
          f"平均 {out['比值_气温相关'].mean():.4f}")
    print(f"控制辐射后与气温的偏相关：{out['控辐射后_气温相关'].min():.4f} ~ "
          f"{out['控辐射后_气温相关'].max():.4f}，平均 {out['控辐射后_气温相关'].mean():.4f}")
    print(f"温度效应（每升高 1 ℃ 的比值变化）：{out['温度效应_每度百分比'].min():.4f} ~ "
          f"{out['温度效应_每度百分比'].max():.4f} %/℃，平均 "
          f"{out['温度效应_每度百分比'].mean():.4f} %/℃")
    print()
    print("【对照】宁夏数据：比值 CV 仅 0.23%，比值与气温相关 +0.0038（几乎为零）")
    print("【判读】若此处 CV 明显更大、且相关系数为负，说明 PVOD 的 power 是真实计量值。")
    print("\n已保存：results/pvod_target_check.csv")


if __name__ == "__main__":
    main()
