#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
河北 PVOD 十站：留一站交叉验证（Leave-One-Station-Out）迁移实验
=============================================================
实验目的：回答「光伏功率预测里，多站预训练再微调到底有没有用」

对每个目标站 H，训练四个模型，在同一个测试集上对比：

  A. 从零训练  ：只用目标站自己的训练数据
  B. 预训练+微调：其余 9 站预训练 -> 用目标站数据继续加树
  C. 零样本迁移 ：只用 9 站预训练模型，完全不看目标站的数据
  D. 多站联合   ：9 站 + 目标站数据混在一起训练（目标站样本加权）

公平性保证：A 与 B 的总树数相同（700 棵），避免"树多所以赢"。

【2026-10-03 口径统一（三处修正，旧结果已作废）】
  1. 时区：PVOD 的时间戳是 UTC，+8 小时才是北京时间。
     原脚本未换算，导致 hour_sin/cos、doy_sin/cos 全部错位 8 小时，
     模型学到的日内模式是错的（详见 code/10 的数据质量检查与 fig10）。
  2. 特征：由 nwp_*（预报值）改为 lmd_*（站内实测值），
     与主线实验 code/11 的设定甲保持一致，否则同一篇论文里
     树模型用实测、迁移实验用预报，两套数字无法横向比较。
  3. 清洗：去掉原来「归一化出力 > 1.05 就删」这一条，
     与 pvod_common.load_station 的清洗规则统一
     （只剔除夜间却仍在发电的记录）。station04 归一化峰值达 1.338，
     原规则会把它整段高值样本删掉，与主线不可比。

用法：
    python3 06_transfer_loso.py            # 跑全部 10 站
    python3 06_transfer_loso.py station01  # 只跑一站（用于快速调试）
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.expanduser('~/Desktop/电气club作业')
DATA = os.path.join(BASE, 'data', 'PVOD_hebei')
RES = os.path.join(BASE, 'results')
os.makedirs(RES, exist_ok=True)

# ---- 实验超参 ----
PRE_ROUNDS = 500      # 预训练树数
FINETUNE_ROUNDS = 200 # 微调时新加的树数
FROM_SCRATCH = PRE_ROUNDS + FINETUNE_ROUNDS   # 700，保证 A 与 B 树数相同
LR = 0.05
LR_FT = 0.02          # 微调步长更小，避免一脚踹偏
TARGET_WEIGHT = 5.0   # 联合训练中目标站样本的权重
TRAIN_RATIO = 0.7     # 目标站内部按时间前 70% 训练 / 后 30% 测试

LMD_FEATS = ['lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature',
             'lmd_pressure', 'lmd_windspeed', 'lmd_winddirection']
STATIC_FEATS = ['cap_norm', 'tilt', 'lat', 'lon']


def add_time_features(df):
    """从时间戳构造周期特征 + 风向的圆投影。

    时间戳已在 load_all() 里统一 +8 小时换算成北京时间，
    这里直接用它算小时，日内模式才对得上（辐射峰值应出现在 12 时前后）。
    """
    t = pd.to_datetime(df['date_time'])
    df = df.copy()
    hour = t.dt.hour + t.dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    doy = t.dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.0)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.0)
    wd = np.deg2rad(df['lmd_winddirection'].fillna(0))
    df['wd_sin'] = np.sin(wd)
    df['wd_cos'] = np.cos(wd)
    return df


TIME_FEATS = ['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'wd_sin', 'wd_cos']
ALL_FEATS = LMD_FEATS + TIME_FEATS


def load_all():
    """读 10 个站 + metadata，做清洗，返回 {站名: DataFrame}"""
    meta = pd.read_csv(os.path.join(DATA, 'metadata.csv'), encoding='utf-8-sig')
    meta['tilt'] = meta['Array_Tilt'].str.extract(r'(\d+)').astype(float)
    data, info = {}, []
    for _, r in meta.iterrows():
        sid = r['Station_ID']
        df = pd.read_csv(os.path.join(DATA, f'{sid}.csv'), encoding='utf-8-sig')
        cap = float(r['Capacity'])

        n0 = len(df)
        # 清洗：夜间却还在发电的记录（辐照为 0 而功率明显不为 0）。
        # 与 pvod_common.load_station 保持同一条规则，不再额外删除超容量样本。
        bad_night = (df['lmd_totalirrad'] <= 0) & (df['power'] > 0.01)
        df['p'] = df['power'] * 1000.0 / cap
        df = df[~bad_night].reset_index(drop=True)

        # 时区：PVOD 原始时间戳是 UTC，+8 小时换算成北京时间
        df['date_time'] = pd.to_datetime(df['date_time']) + pd.Timedelta(hours=8)

        df = add_time_features(df)
        df['cap_norm'] = cap / 35000.0       # 容量归一化，最大站为 1
        df['tilt'] = float(r['tilt'])
        df['lat'] = float(r['Latitude'])
        df['lon'] = float(r['Longitude'])
        df['cap'] = cap
        data[sid] = df
        info.append(dict(站=sid, 容量kW=cap, 原始行数=n0, 清洗后=len(df),
                         清洗率=round((n0 - len(df)) / n0 * 100, 2),
                         起=str(df['date_time'].min())[:10],
                         止=str(df['date_time'].max())[:10]))
    return meta, data, pd.DataFrame(info)


def metrics(y_true, y_pred, day_mask=None):
    """MAE / RMSE，单位 %装机容量"""
    e = np.asarray(y_pred) - np.asarray(y_true)
    out = {'MAE': float(np.abs(e).mean()) * 100,
           'RMSE': float(np.sqrt((e ** 2).mean())) * 100}
    if day_mask is not None and day_mask.sum() > 0:
        ed = e[day_mask]
        out['MAE白天'] = float(np.abs(ed).mean()) * 100
        out['RMSE白天'] = float(np.sqrt((ed ** 2).mean())) * 100
    return out


def main():
    try:
        import lightgbm as lgb
    except ImportError:
        print('缺少 lightgbm，请先执行： pip install lightgbm')
        return 1

    meta, data, info = load_all()
    stations = sorted(data.keys())
    only = sys.argv[1] if len(sys.argv) > 1 else None
    targets = [only] if only else stations

    print('=' * 78)
    print('数据加载与清洗结果')
    print('=' * 78)
    print(info.to_string(index=False))

    base_params = dict(objective='regression', metric='l2', learning_rate=LR,
                       num_leaves=31, min_data_in_leaf=20,
                       feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
                       lambda_l2=1.0, verbose=-1, seed=42, num_threads=4)

    rows, detail = [], []

    for tgt in targets:
        src = [s for s in stations if s != tgt]
        df_t = data[tgt].sort_values('date_time').reset_index(drop=True)

        # 目标站内部按时间切分：前 70% 训练，后 30% 测试
        cut = int(len(df_t) * TRAIN_RATIO)
        tr_t, te_t = df_t.iloc[:cut], df_t.iloc[cut:]
        day_te = (te_t['lmd_totalirrad'] > 0).values

        dtrain_t = lgb.Dataset(tr_t[ALL_FEATS], label=tr_t['p'], free_raw_data=False)
        X_te, y_te = te_t[ALL_FEATS], te_t['p'].values

        # ---- 0) 源站预训练：其余 9 站混合 ----
        df_src = pd.concat([data[s] for s in src], ignore_index=True)
        dsrc = lgb.Dataset(df_src[ALL_FEATS], label=df_src['p'], free_raw_data=False)
        pre = lgb.train(base_params, dsrc, num_boost_round=PRE_ROUNDS)

        # ---- C) 零样本：预训练模型直接预测目标站（不看目标站任何标签）----
        m_c = metrics(y_te, pre.predict(X_te), day_te)

        # ---- A) 从零训练：只用目标站训练数据，树数与 B 对齐 ----
        pa = {**base_params, 'learning_rate': LR}
        m_a_model = lgb.train(pa, dtrain_t, num_boost_round=FROM_SCRATCH)
        m_a = metrics(y_te, m_a_model.predict(X_te), day_te)

        # ---- B) 预训练 + 微调：接着加树，步长更小 ----
        pb = {**base_params, 'learning_rate': LR_FT}
        m_b_model = lgb.train(pb, dtrain_t, num_boost_round=FINETUNE_ROUNDS,
                              init_model=pre)
        m_b = metrics(y_te, m_b_model.predict(X_te), day_te)

        # ---- D) 多站联合训练：源全部 + 目标站训练集（目标站样本加权）----
        joint = pd.concat([df_src, tr_t], ignore_index=True)
        w = np.where(joint['cap'] == df_t['cap'].iloc[0], TARGET_WEIGHT, 1.0)
        feats_j = ALL_FEATS + STATIC_FEATS
        dj = lgb.Dataset(joint[feats_j], label=joint['p'], weight=w, free_raw_data=False)
        X_te_j = te_t[feats_j]
        m_d_model = lgb.train(base_params, dj, num_boost_round=PRE_ROUNDS)
        m_d = metrics(y_te, m_d_model.predict(X_te_j), day_te)

        rec = dict(站=tgt, 容量MW=df_t['cap'].iloc[0] / 1000,
                   测试样本=len(te_t), 白天样本=int(day_te.sum()))
        # 两套口径都存：全天 / 仅白天。报告里引用时必须标明是哪一个，不能混用。
        for tag, m in [('A从零', m_a), ('B微调', m_b), ('C零样本', m_c), ('D联合', m_d)]:
            rec[f'{tag}_MAE'] = round(m['MAE'], 3)
            rec[f'{tag}_RMSE'] = round(m['RMSE'], 3)
            rec[f'{tag}_MAE白天'] = round(m.get('MAE白天', np.nan), 3)
            rec[f'{tag}_RMSE白天'] = round(m.get('RMSE白天', np.nan), 3)
        # 提升率：两个口径分别计算（相对 A 的 MAE 降幅，正=变好）
        for suffix in ['', '白天']:
            for tag in ['B微调', 'C零样本', 'D联合']:
                a = rec[f'A从零_MAE{suffix}']
                v = rec[f'{tag}_MAE{suffix}']
                rec[f'{tag}较A提升{suffix}%'] = round((a - v) / a * 100, 2)
        rows.append(rec)

        print(f'[{tgt}] A={m_a["MAE"]:.3f}  B={m_b["MAE"]:.3f}  '
              f'C={m_c["MAE"]:.3f}  D={m_d["MAE"]:.3f}  (%容量)')

    out = pd.DataFrame(rows)
    print()
    print('=' * 78)
    print('留一站交叉验证汇总（MAE / RMSE 单位 = %装机容量，越低越好）')
    print('=' * 78)
    show_cols = ['站', '容量MW', '测试样本', '白天样本',
                 'A从零_MAE', 'B微调_MAE', 'C零样本_MAE', 'D联合_MAE',
                 'A从零_MAE白天', 'B微调_MAE白天', 'C零样本_MAE白天', 'D联合_MAE白天']
    print(out[show_cols].to_string(index=False))

    print()
    print('-' * 78)
    print('两套口径的提升率（务必在报告中写明用的是哪一个）')
    print('-' * 78)
    print('【口径一：全天样本，含夜间零功率】')
    for tag in ['B微调', 'C零样本', 'D联合']:
        v = out[f'{tag}较A提升%']
        print(f'  {tag:<8s} 平均 {v.mean():>7.2f}%   赢过 A 的站数 {int((v > 0).sum()):>2d}/{len(out)}')
    print('【口径二：仅白天样本，辐照 > 0】')
    for tag in ['B微调', 'C零样本', 'D联合']:
        v = out[f'{tag}较A提升白天%']
        print(f'  {tag:<8s} 平均 {v.mean():>7.2f}%   赢过 A 的站数 {int((v > 0).sum()):>2d}/{len(out)}')
    print()
    print('  平均 MAE / RMSE：')
    for tag in ['A从零', 'B微调', 'C零样本', 'D联合']:
        print(f'    {tag:<8s} MAE={out[f"{tag}_MAE"].mean():.3f}  RMSE={out[f"{tag}_RMSE"].mean():.3f}'
              f'   | 白天 MAE={out[f"{tag}_MAE白天"].mean():.3f}  RMSE={out[f"{tag}_RMSE白天"].mean():.3f}')

    csv_all = os.path.join(RES, 'transfer_loso_results.csv')
    csv_info = os.path.join(RES, 'transfer_loso_datainfo.csv')
    out.to_csv(csv_all, index=False, encoding='utf-8-sig')
    info.to_csv(csv_info, index=False, encoding='utf-8-sig')
    print()
    print('已保存:', csv_all)
    print('已保存:', csv_info)
    return 0


if __name__ == '__main__':
    sys.exit(main())
