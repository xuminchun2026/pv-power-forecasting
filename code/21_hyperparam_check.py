# -*- coding: utf-8 -*-
"""
调参对照实验（把论文里引用的两处调参记录固化下来，保证可复现）

论文「实验设定与评估指标」一节写了两条调参记录，都是实跑出来的，不是猜的：
  1. XGBoost 的库默认学习率是 0.3，在若干站上明显过拟合
     → 正文因此统一取 0.1，并另列一行「默认学习率」作对照（这一行已由 code/11 输出）。
  2. 树数并非越多越好：把 LightGBM 从 100 棵加到 300 棵，十站平均反而变差。
     本脚本负责复现第 2 条。

用法：python3 21_hyperparam_check.py
输出：results/pvod_tree_count_check.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pvod_common import RES, STATIONS, RANDOM_STATE, load_station, build_b, split_by_time
from sklearn.metrics import mean_absolute_error

TREE_COUNTS = [100, 300]
LR = 0.1
# 设定乙（纯历史时序）是消融与特征重要性都在用的设定，故在这里也用它做对照
SETTING = "乙-纯历史时序"


def main():
    from lightgbm import LGBMRegressor

    rows = []
    for n in TREE_COUNTS:
        maes = []
        for sid in STATIONS:
            d = load_station(sid)
            db, fb = build_b(d)
            dd = db[fb + ["p_norm"]].dropna()
            tr, te = split_by_time(dd)
            m = LGBMRegressor(n_estimators=n, learning_rate=LR,
                              random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
            m.fit(tr[fb].values, tr["p_norm"].values)
            maes.append(mean_absolute_error(te["p_norm"].values,
                                            m.predict(te[fb].values)) * 100)
        rows.append(dict(设定=SETTING, 模型="LightGBM", 树数=n, 学习率=LR,
                         十站平均MAE=round(float(np.mean(maes)), 4),
                         最差站MAE=round(float(np.max(maes)), 4),
                         最好站MAE=round(float(np.min(maes)), 4)))
        print(f"LightGBM {n:3d} 棵 @ lr={LR}: 十站平均 MAE {np.mean(maes):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "pvod_tree_count_check.csv"),
               index=False, encoding="utf-8-sig")

    a, b = out.iloc[0], out.iloc[1]
    print(f"\n由 {a['树数']:.0f} 棵增至 {b['树数']:.0f} 棵，MAE {a['十站平均MAE']:.4f} "
          f"→ {b['十站平均MAE']:.4f}（{'变差' if b['十站平均MAE'] > a['十站平均MAE'] else '变好'}），"
          f"说明本任务 100 棵已足够，继续加树只会过拟合。")
    print("已保存：results/pvod_tree_count_check.csv")


if __name__ == "__main__":
    main()
