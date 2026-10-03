#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
河北 PVOD 十站：LSTM 的「全局预训练 + 单站微调」实验
==================================================
与 06_transfer_loso.py（LightGBM 版）一一对应，便于横向比较：

  A. 从零训练   ：只用目标站自己的数据
  B. 预训练+微调：其余 9 站预训练 -> 目标站数据微调（这是 LSTM 真正意义的微调）
  C. 零样本迁移 ：9 站预训练模型直接套到目标站
  D. 多站联合   ：9 站 + 目标站混合训练（目标站样本权重 ×5，输入带站点静态属性）
  E. 预训练初始化 + 目标站训满：与 A 训练量完全相同，只有初始权重不同

与树模型的关键差别：
  树模型的"微调"只是接着加树，老树改不掉；
  LSTM 的微调是**继续反向传播，所有权重都会跟着动**——这正是本实验要检验的。

公平性：
  A 与 E 轮数相同（都是 SCRATCH_EPOCHS），B 的总轮数 = PRE + FT；
  B 与 E 共用同一份预训练权重（deepcopy），起点完全一致；
  D 与 LightGBM 版一样额外拿到 4 个站点静态属性，否则它分不清自己在给哪个站建模。

任务定义：用过去 SEQ_LEN 步（默认 12 步 = 3 小时）的 nwp 预报值与时间特征，
预测当前时刻的归一化功率（功率/装机容量）。

用法：
    python3 08_lstm_transfer.py            # 跑全部 10 站
    python3 08_lstm_transfer.py station01  # 只跑一站（调试用）
"""

import os
import sys
import time
import copy
import importlib.util
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.expanduser('~/Desktop/电气club作业')
RES = os.path.join(BASE, 'results')
os.makedirs(RES, exist_ok=True)

# 复用 06 号脚本里的读数据与清洗逻辑（它带 main guard，导入不会执行实验）
_spec = importlib.util.spec_from_file_location(
    'loso', os.path.join(os.path.dirname(os.path.abspath(__file__)), '06_transfer_loso.py'))
loso = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loso)

# ---- 超参 ----
SEQ_LEN = 12          # 回看窗口：12 × 15 min = 3 小时
PRE_EPOCHS = 10       # 预训练轮数
FT_EPOCHS = 5         # 微调轮数
SCRATCH_EPOCHS = PRE_EPOCHS + FT_EPOCHS   # 15，保证 A 与 B/D 训练量相同
HIDDEN = 64
NLAYERS = 2
DROPOUT = 0.1
BATCH = 2048
LR = 1e-3
LR_FT = 3e-4          # 微调步长更小
TARGET_WEIGHT = 5.0
TRAIN_RATIO = 0.7
SEED = 42

FEATS = loso.ALL_FEATS                              # 时序特征 13 个：7 个 nwp + 6 个时间
STATIC = loso.STATIC_FEATS                          # 站点静态属性 4 个：容量/倾角/纬度/经度
FEATS_J = FEATS + STATIC                            # 联合训练用 17 个（与 LightGBM 版 D 对齐）
NF, NJ = len(FEATS), len(FEATS_J)


# ------------------------------------------------------------------
# 数据：把每站切成 (窗口 X, 目标 y, 白天掩码) 三段
# ------------------------------------------------------------------
def make_windows(df, seq=SEQ_LEN):
    """滑动窗口。X[k] 对应时刻 t=k+seq-1，只看过去，不看未来。"""
    X = df[FEATS_J].values.astype(np.float32)
    y = df['p'].values.astype(np.float32)
    day = (df['lmd_totalirrad'].values > 0)
    n = len(df)
    # 注意：sliding_window_view 在二维输入上返回的是 (n-seq+1, 特征, seq)，
    # 时间轴在最后一维；要转置成 LSTM 习惯的 (样本, seq, 特征)。
    W = np.lib.stride_tricks.sliding_window_view(X, seq, axis=0)
    W = np.ascontiguousarray(W).transpose(0, 2, 1)
    yy = y[seq - 1:]
    dd = day[seq - 1:]
    t_idx = np.arange(seq - 1, n)                                  # 每个窗口的目标时刻
    return W, yy, dd, t_idx


def split_by_time(W, y, day, t_idx, n_total, ratio=TRAIN_RATIO):
    """按时间切分：目标时刻在前 70% 的窗口用于训练。窗口只往回看，故无未来泄漏。"""
    cut = int(n_total * ratio)
    tr = t_idx < cut
    te = ~tr
    return (W[tr], y[tr], day[tr]), (W[te], y[te], day[te])


# ------------------------------------------------------------------
# 模型
# ------------------------------------------------------------------
def build_model(nfeat, device):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(nfeat, HIDDEN, NLAYERS, batch_first=True,
                                dropout=DROPOUT if NLAYERS > 1 else 0.0)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1, :]).squeeze(-1)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return Net().to(device)


def fit(model, X, y, device, epochs, lr, w=None, tag=''):
    """训练。w 为逐样本权重，直接乘进损失里，不靠重复采样，省内存"""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    if w is None:
        w = np.ones(len(X), dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(w))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss(reduction='none')
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(xb)
        if tag:
            print(f'      {tag} epoch {ep+1}/{epochs}  loss={tot/len(ds):.5f}')
    if device == 'mps':
        torch.mps.empty_cache()
    return model


def predict(model, X, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i:i + 4096]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def metrics(y_true, y_pred, day_mask=None):
    e = np.asarray(y_pred) - np.asarray(y_true)
    out = {'MAE': float(np.abs(e).mean()) * 100,
           'RMSE': float(np.sqrt((e ** 2).mean())) * 100}
    if day_mask is not None and day_mask.sum() > 0:
        ed = e[day_mask]
        out['MAE白天'] = float(np.abs(ed).mean()) * 100
        out['RMSE白天'] = float(np.sqrt((ed ** 2).mean())) * 100
    return out


# ------------------------------------------------------------------
def main():
    try:
        import torch
    except ImportError:
        print('缺少 torch，请先执行： pip install torch')
        return 1

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('计算设备:', device, '| torch', torch.__version__)

    _, data, info = loso.load_all()
    stations = sorted(data.keys())
    only = sys.argv[1] if len(sys.argv) > 1 else None
    targets = [only] if only else stations

    print('=' * 78)
    print(f'LSTM 迁移实验  窗口={SEQ_LEN}步({SEQ_LEN*15}分钟)  '
          f'预训练{PRE_EPOCHS}轮 + 微调{FT_EPOCHS}轮  从零{SCRATCH_EPOCHS}轮')
    print('=' * 78)

    # 预先构造各站窗口（17 列 = 13 时序 + 4 静态）
    W_all = {}
    for sid in stations:
        df = data[sid]
        W, y, day, t_idx = make_windows(df)
        (Xtr, ytr, dtr), (Xte, yte, dte) = split_by_time(W, y, day, t_idx, len(df))
        # 时序部分的标准化统计量：统一用目标站训练段，保证各方案在同一尺度上
        mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
        sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
        W_all[sid] = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                          dte=dte, mu=mu, sd=sd, nte=len(yte))

    # 静态属性只有 10 个取值（一站一个），用跨站的均值和标准差来缩放，
    # 这样 10 个站在这一维上才真的能区分开；用单站统计量会把它们全缩成 0。
    st = np.array([[data[s][c].iloc[0] for c in STATIC] for s in stations], dtype=np.float64)
    mu_s, sd_s = st.mean(0), st.std(0) + 1e-8
    print('静态属性缩放（跨 10 站）：')
    for c, m_, s_ in zip(STATIC, mu_s, sd_s):
        print(f'  {c:<10s} 均值 {m_:.3f}  标准差 {s_:.3f}')

    def norm(sid, X):
        """标准化：前 13 列按目标站训练段，后 4 列按跨站统计量"""
        d = W_all[sid]
        mu = d['mu'].copy()
        sd = d['sd'].copy()
        mu[NF:] = mu_s
        sd[NF:] = sd_s
        return ((X - mu) / sd).astype(np.float32)

    rows = []
    t_all = time.time()
    for tgt in targets:
        d = W_all[tgt]
        Xtr_t, ytr_t = norm(tgt, d['Xtr'])[:, :, :NF], d['ytr']
        Xte_t, yte_t = norm(tgt, d['Xte'])[:, :, :NF], d['yte']
        # 联合训练用完整 17 列（多出来的 4 列让模型知道"这是哪个站"）
        Xtr_j, Xte_j = norm(tgt, d['Xtr']), norm(tgt, d['Xte'])
        src = [s for s in stations if s != tgt]

        # 源站数据：各站用自己的训练段（避免用到别的站的未来）
        Xs = np.concatenate([norm(tgt, W_all[s]['Xtr'])[:, :, :NF] for s in src])
        Xs_j = np.concatenate([norm(tgt, W_all[s]['Xtr']) for s in src])
        ys = np.concatenate([W_all[s]['ytr'] for s in src])
        print(f'\n--- 目标站 {tgt}  训练样本 {len(ytr_t)}  测试样本 {len(yte_t)}  源站样本 {len(ys)}')

        nfeat = Xtr_t.shape[-1]

        # A 从零训练
        print('  [A] 从零训练')
        mA = build_model(nfeat, device)
        fit(mA, Xtr_t, ytr_t, device, SCRATCH_EPOCHS, LR)
        m_a = metrics(yte_t, predict(mA, Xte_t, device), d['dte'])
        del mA

        # 先做一次预训练，B / C / E 共用这一份权重
        # （B 和 E 的唯一差别是后面怎么训，起点必须完全一样，否则对照不干净）
        print('  [预训练] 9 站 %d 轮' % PRE_EPOCHS)
        mP = build_model(nfeat, device)
        fit(mP, Xs, ys, device, PRE_EPOCHS, LR)

        # C 零样本：预训练完直接拿到目标站上测
        m_c = metrics(yte_t, predict(mP, Xte_t, device), d['dte'])

        # B 预训练 + 微调（真实微调场景：小步长、少轮数）
        print('  [B] 预训练 -> 目标站微调（小步长 %d 轮）' % FT_EPOCHS)
        mB = copy.deepcopy(mP)
        fit(mB, Xtr_t, ytr_t, device, FT_EPOCHS, LR_FT)
        m_b = metrics(yte_t, predict(mB, Xte_t, device), d['dte'])
        del mB

        # E 预训练初始化 + 与 A 完全相同的训练量（最干净的对照）
        # 与 A 的唯一差别是初始权重：A 随机初始化，E 用源站预训练权重。
        # 若 E 优于 A，说明预训练带来的初始化本身有价值；若 E≈A，说明价值只来自"多看了源站数据"。
        print('  [E] 预训练初始化 -> 目标站训满 %d 轮' % SCRATCH_EPOCHS)
        mE = copy.deepcopy(mP)
        fit(mE, Xtr_t, ytr_t, device, SCRATCH_EPOCHS, LR)
        m_e = metrics(yte_t, predict(mE, Xte_t, device), d['dte'])
        del mE, mP

        # D 多站联合（目标站样本权重 ×5），输入带站点静态属性
        print('  [D] 多站联合训练（目标站权重 ×5，含站点属性）')
        Xj = np.concatenate([Xs_j, Xtr_j])
        yj = np.concatenate([ys, ytr_t])
        wj = np.concatenate([np.ones(len(ys), dtype=np.float32),
                             np.full(len(ytr_t), TARGET_WEIGHT, dtype=np.float32)])
        mD = build_model(NJ, device)
        fit(mD, Xj, yj, device, SCRATCH_EPOCHS, LR, w=wj)
        m_d = metrics(yte_t, predict(mD, Xte_j, device), d['dte'])
        del mD

        rec = dict(站=tgt, 测试样本=len(yte_t))
        for tag, m in [('A从零', m_a), ('B微调', m_b), ('C零样本', m_c),
                       ('D联合', m_d), ('E预训练初始化', m_e)]:
            rec[f'{tag}_MAE'] = round(m['MAE'], 3)
            rec[f'{tag}_RMSE'] = round(m['RMSE'], 3)
            rec[f'{tag}_MAE白天'] = round(m['MAE白天'], 3)
        base = m_a['MAE']
        for tag, m in [('B微调', m_b), ('C零样本', m_c),
                       ('D联合', m_d), ('E预训练初始化', m_e)]:
            rec[f'{tag}较A提升%'] = round((base - m['MAE']) / base * 100, 2)
        rows.append(rec)
        print(f'      A={m_a["MAE"]:.3f}  B={m_b["MAE"]:.3f}({rec["B微调较A提升%"]:+.2f}%)  '
              f'C={m_c["MAE"]:.3f}  D={m_d["MAE"]:.3f}({rec["D联合较A提升%"]:+.2f}%)  '
              f'E={m_e["MAE"]:.3f}({rec["E预训练初始化较A提升%"]:+.2f}%)')

    out = pd.DataFrame(rows)
    print('\n' + '=' * 78)
    print('LSTM 留一站交叉验证汇总（MAE / RMSE 单位 = %装机容量，越低越好）')
    print('=' * 78)
    print(out.to_string(index=False))
    print('\n平均：')
    for tag in ['A从零', 'B微调', 'C零样本', 'D联合', 'E预训练初始化']:
        print(f'  {tag:<14s} MAE = {out[f"{tag}_MAE"].mean():.3f}   '
              f'RMSE = {out[f"{tag}_RMSE"].mean():.3f}   '
              f'MAE白天 = {out[f"{tag}_MAE白天"].mean():.3f}')
    for tag in ['B微调', 'C零样本', 'D联合', 'E预训练初始化']:
        col = f'{tag}较A提升%'
        print(f'  {tag:<14s} 平均提升 {out[col].mean():+.2f}%   赢过从零的站数 {(out[col] > 0).sum()}/{len(out)}')

    csv = os.path.join(RES, 'lstm_transfer_results.csv')
    out.to_csv(csv, index=False, encoding='utf-8-sig')
    print('\n已保存:', csv)
    print(f'总耗时 {time.time()-t_all:.0f} 秒')
    return 0


if __name__ == '__main__':
    sys.exit(main())
