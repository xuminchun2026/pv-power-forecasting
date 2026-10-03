# -*- coding: utf-8 -*-
"""
PVOD 河北十站 —— 单站 LSTM 对照实验（任务清单核心第 3 项）

【要回答的问题】
  研究问题 1「复杂模型是否一定优于简单模型」在深度模型这一侧是否也成立：
  LSTM 在单站内的四种信息条件下，能否超过朴素基线和树模型？

【四个设定，与 code/11 的树模型严格对齐】
  甲  同时刻站内实测气象 (lmd_*)          —— 序列结束于时刻 t
  甲′ 同时刻数值天气预报 (nwp_*)           —— 序列结束于时刻 t
  乙  纯历史观测（无任何 t 时刻信息）      —— 序列结束于时刻 t-1，并把历史出力放进序列
  丙  同时刻气象 + 历史出力                —— 气象用 t，出力序列用到 t-1

【与树模型可比的三条保证】
  1. 同一份数据、同一条前 70% / 后 30% 的时间切分（不打乱）。
  2. 信息可得性完全一致：树模型设定乙用 safe_lag 只取历史，
     LSTM 设定乙的窗口同样只覆盖到 t-1，绝不偷看 t。
  3. 断点处理一致：树模型在断点处滞后值为 NaN 后被 dropna；
     本脚本剔除任何跨越时间断点的窗口，避免把「几天前的同一时刻」当成连续历史。

【为什么 LSTM 用序列而树模型用滞后特征】
  树模型设定乙的输入是 GHI_lag1/2/3/6/12、P_lag1、T_lag1、T_lag6，
  LSTM 的输入是过去 12 步（3 小时）逐点的多变量序列。
  两者看到的时间跨度都是 3 小时左右，信息量相当，只是组织方式不同，
  这正是本实验要比较的东西。
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pvod_common import (RES, STATIONS, GHI, NWP_GHI, LMD_FEATS, NWP_FEATS,
                         TIME_FEATS, STEP, load_station, add_time_features,
                         split_by_time)

SEQ_LEN = 12      # 窗口长度：12 步 × 15 min = 过去 3 小时
EPOCHS = 30       # 固定 30 轮（未做早停与调参），给足训练量以免被质疑「没训够」
BATCH = 2048
HIDDEN = 64
NLAYERS = 2
LR = 1e-3
SEED = 42

SETTINGS = ["甲-同时刻实测(lmd)", "甲-同时刻预报(nwp)", "乙-纯历史时序", "丙-同时刻气象+历史"]


# ------------------------------------------------------------------
# 窗口构造
# ------------------------------------------------------------------
def make_windows(d, met_cols, with_p, end_at_now):
    """滑动窗口。

    参数
    ----
    met_cols   : 气象变量列名（不含时刻编码）
    with_p     : 是否把历史归一化出力 p_norm 放进序列
    end_at_now : True  → 气象序列覆盖 t-SEQ+1 .. t（同时刻气象可得）
                 False → 气象序列覆盖 t-SEQ   .. t-1（只用历史）

    返回 X (n, SEQ, nfeat)、y (n,)、以及每个窗口的目标行号 t_idx
    """
    cols = met_cols + TIME_FEATS
    M = d[cols].values.astype(np.float32)
    P = d["p_norm"].values.astype(np.float32)
    y = d["p_norm"].values.astype(np.float32)
    n = len(d)

    # 气象窗口：起点 s_met，终点 s_met+SEQ-1
    #   end_at_now 时终点 = t → s_met = t-SEQ+1
    #   否则终点 = t-1  → s_met = t-SEQ
    off = SEQ_LEN - 1 if end_at_now else SEQ_LEN
    # 出力序列永远只到 t-1：起点 s_p = t-SEQ
    sp0 = SEQ_LEN

    n_win = n - SEQ_LEN          # t 从 SEQ_LEN 开始（保证 t-SEQ >= 0）
    t_idx = np.arange(SEQ_LEN, n)

    idx_met = t_idx[:, None] - off + np.arange(SEQ_LEN)[None, :]     # (nwin, SEQ)
    Wm = M[idx_met]                                                  # (nwin, SEQ, nfeat_met)

    if with_p:
        idx_p = t_idx[:, None] - sp0 + np.arange(SEQ_LEN)[None, :]   # t-SEQ .. t-1
        Wp = P[idx_p][:, :, None]                                    # (nwin, SEQ, 1)
        X = np.concatenate([Wm, Wp], axis=2)
    else:
        X = Wm

    # ---- 剔除跨越时间断点的窗口 ----
    # 只有窗口覆盖区间内每一步相邻间隔都恰好是 15 min 才保留
    ok = (d["date_time"].diff() == STEP).values.astype(np.int32)
    c = np.concatenate([[0], np.cumsum(ok)])
    lo = t_idx - SEQ_LEN + 1          # 窗口最左端
    span_ok = c[t_idx] - c[lo]        # 区间内相邻间隔正常的个数
    keep = span_ok == SEQ_LEN - 1

    return X[keep], y[t_idx][keep], t_idx[keep], int(keep.sum()), int(n_win)


# ------------------------------------------------------------------
# 模型
# ------------------------------------------------------------------
def build_model(nfeat, device):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(nfeat, HIDDEN, NLAYERS, batch_first=True,
                                dropout=0.0)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return Net().to(device)


def fit(model, X, y, device, epochs=EPOCHS):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if device == "mps":
            torch.mps.empty_cache()
    return time.time() - t0


def predict(model, X, device):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i:i + 4096]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return (mean_absolute_error(y_true, y_pred) * 100,
            np.sqrt(mean_squared_error(y_true, y_pred)) * 100,
            r2_score(y_true, y_pred))


# ------------------------------------------------------------------
def main():
    try:
        import torch
    except ImportError:
        print("缺少 torch，请先执行： pip install torch")
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("计算设备:", device, "| torch", torch.__version__)
    print(f"窗口={SEQ_LEN}步({SEQ_LEN*15}分钟)  轮数={EPOCHS}  隐藏层={HIDDEN}×{NLAYERS}")
    print("=" * 84)

    rows = []
    for sid in STATIONS:
        d = load_station(sid)
        d = add_time_features(d)
        n_total = len(d)
        cut = int(n_total * 0.7)          # 与 pvod_common.split_by_time 同一条线

        plans = [
            ("甲-同时刻实测(lmd)", LMD_FEATS, False, True),
            ("甲-同时刻预报(nwp)", NWP_FEATS, False, True),
            ("乙-纯历史时序",      LMD_FEATS, True,  False),
            ("丙-同时刻气象+历史", LMD_FEATS, True,  True),
        ]

        line = [f"{sid}"]
        for tag, met, with_p, now in plans:
            X, y, t_idx, n_keep, n_all = make_windows(d, met, with_p, now)
            tr = t_idx < cut
            te = ~tr
            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]

            model = build_model(X.shape[2], device)
            sec = fit(model, Xtr, ytr, device)
            pred = predict(model, Xte, device)
            mae, rmse, r2 = metrics(yte, pred)

            rows.append(dict(station=sid, 设定=tag, 方法="LSTM",
                             MAE=round(mae, 4), RMSE=round(rmse, 4), R2=round(r2, 4),
                             训练秒=round(sec, 2),
                             n_train=int(tr.sum()), n_test=int(te.sum()),
                             剔除断点窗口=int(n_all - n_keep)))
            line.append(f"{tag.split('-')[0]}={mae:.4f}")

            del model
            if device == "mps":
                torch.mps.empty_cache()

        print(" | ".join(line))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "pvod_lstm_single.csv"),
               index=False, encoding="utf-8-sig")

    print("\n" + "=" * 84)
    print("单站 LSTM 十站平均（MAE / RMSE 单位 %装机容量）")
    avg = (out.groupby("设定", as_index=False)
              .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"),
                   R2=("R2", "mean"), 训练秒=("训练秒", "mean")))
    avg = avg.set_index("设定").loc[SETTINGS].reset_index()
    print(avg.round(4).to_string(index=False))

    # ---- 与树模型合并比较 ----
    try:
        trees = pd.read_csv(os.path.join(RES, "pvod_main_per_station.csv"))
        keep_methods = ["朴素基线: 上一时刻功率(Persistence)",
                        "朴素基线: 固定系数(P=k·GHI)",
                        "XGBoost", "LightGBM", "随机森林RF", "线性回归"]
        t = trees[trees["方法"].isin(keep_methods)]
        merged = pd.concat([t[["station", "设定", "方法", "MAE", "RMSE", "R2"]],
                            out[["station", "设定", "方法", "MAE", "RMSE", "R2"]]],
                           ignore_index=True)
        cmp = (merged.groupby(["设定", "方法"], as_index=False)
                     .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), R2=("R2", "mean")))
        cmp.to_csv(os.path.join(RES, "pvod_lstm_vs_trees.csv"),
                   index=False, encoding="utf-8-sig")
        print("\n各设定下 LSTM 与树模型十站平均对比（MAE %装机容量）")
        piv = cmp.pivot(index="设定", columns="方法", values="MAE")
        piv = piv.reindex(SETTINGS)
        print(piv.round(4).to_string())
    except FileNotFoundError:
        print("\n（未找到 pvod_main_per_station.csv，合并比较跳过）")

    print("\n已保存：results/pvod_lstm_single.csv")
    print("已保存：results/pvod_lstm_vs_trees.csv")


if __name__ == "__main__":
    main()
