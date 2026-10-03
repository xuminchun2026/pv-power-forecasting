#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把留一站交叉验证的结果画成图，并分析「为什么 station06 会负迁移」
产出：results/transfer_loso_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti SC', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.expanduser('~/Desktop/电气club作业')
RES = os.path.join(BASE, 'results')
DATA = os.path.join(BASE, 'data', 'PVOD_hebei')

res = pd.read_csv(os.path.join(RES, 'transfer_loso_results.csv'), encoding='utf-8-sig')
info = pd.read_csv(os.path.join(RES, 'transfer_loso_datainfo.csv'), encoding='utf-8-sig')

# ---------- 站点间地理距离（判断是否"孤立站"） ----------
meta = pd.read_csv(os.path.join(DATA, 'metadata.csv'), encoding='utf-8-sig')
coord = meta.set_index('Station_ID')[['Latitude', 'Longitude']]
rows = []
for sid in res['站']:
    la, lo = coord.loc[sid]
    d = []
    for o in coord.index:
        if o == sid:
            continue
        la2, lo2 = coord.loc[o]
        dx = (lo - lo2) * 111.0 * np.cos(np.deg2rad(la))
        dy = (la - la2) * 111.0
        d.append(np.hypot(dx, dy))
    rows.append(dict(站=sid, 最近站距离km=round(min(d), 1), 平均站距km=round(np.mean(d), 1)))
geo = pd.DataFrame(rows)
res = res.merge(geo, on='站')

# ---------- 画图 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
labels = [s.replace('station', 'S') for s in res['站']]
x = np.arange(len(labels))
w = 0.2

ax = axes[0]
for i, (tag, c) in enumerate([('A从零', '#7F7F7F'), ('B微调', '#1D9E75'),
                              ('C零样本', '#BA7517'), ('D联合', '#378ADD')]):
    v = res[f'{tag}_MAE'].values
    ax.bar(x + (i - 1.5) * w, v, w, label=tag.replace('_MAE', ''), color=c,
           edgecolor='white', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('MAE（占装机容量 %）')
ax.set_title('(a) 各站点四种方法的 MAE 对比', fontsize=13, pad=10)
ax.legend(fontsize=10, frameon=False, ncol=2)
ax.grid(axis='y', alpha=0.25, linestyle='--')
ax.set_axisbelow(True)

ax = axes[1]
up_b = res['B微调较A提升%'].values
up_d = res['D联合较A提升%'].values
ax.bar(x - w / 2, up_b, w, label='预训练+微调 (B)', color='#1D9E75')
ax.bar(x + w / 2, up_d, w, label='多站联合 (D)', color='#378ADD')
ax.axhline(0, color='#444', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('相对「从零训练」的 MAE 降幅（%）')
ax.set_title('(b) 各站点的提升率（负值=迁移反而变差）', fontsize=13, pad=10)
ax.legend(fontsize=10, frameon=False)
ax.grid(axis='y', alpha=0.25, linestyle='--')
ax.set_axisbelow(True)
# 标出负迁移
for i, v in enumerate(up_b):
    ax.text(i - w / 2, v - (3.0 if v >= 0 else -3.0), f'{v:.0f}',
            ha='center', va='top' if v < 0 else 'bottom', fontsize=9,
            color='#A32D2D' if v < 0 else '#0F6E56')
for i, v in enumerate(up_d):
    ax.text(i + w / 2, v + 0.8, f'{v:.0f}', ha='center', va='bottom', fontsize=9, color='#185FA5')

plt.tight_layout()
png = os.path.join(RES, 'transfer_loso_comparison.png')
plt.savefig(png, dpi=180, bbox_inches='tight')
print('已保存:', png)

# ---------- 分析：负迁移站与其它站的差异 ----------
print()
print('=' * 78)
print('站点地理分布与提升率对照（用于解释负迁移）')
print('=' * 78)
print(res[['站', '容量MW', '最近站距离km', '平均站距km', 'B微调较A提升%', 'D联合较A提升%']].to_string(index=False))
print()
print('结论提示：')
bad = res[res['B微调较A提升%'] < 0]
if len(bad):
    for _, r in bad.iterrows():
        print(f'  ⚠ {r["站"]} 出现负迁移（B 下降 {abs(r["B微调较A提升%"]):.1f}%），'
              f'它到最近邻站 {r["最近站距离km"]} km、平均站距 {r["平均站距km"]} km')
        print(f'     但该站的「多站联合」方案仍为 {r["D联合较A提升%"]:+.1f}%，说明加入本站数据即可纠偏')
print()
print(f'  B 与 D 的比较：D 在 {int((res["D联合较A提升%"] > res["B微调较A提升%"]).sum())} / {len(res)} 个站上优于 B')

# ---------- 汇总关键数字（写报告用） ----------
summ = dict(
    n=len(res),
    A_MAE=round(res['A从零_MAE'].mean(), 3), B_MAE=round(res['B微调_MAE'].mean(), 3),
    C_MAE=round(res['C零样本_MAE'].mean(), 3), D_MAE=round(res['D联合_MAE'].mean(), 3),
    A_RMSE=round(res['A从零_RMSE'].mean(), 3), B_RMSE=round(res['B微调_RMSE'].mean(), 3),
    C_RMSE=round(res['C零样本_RMSE'].mean(), 3), D_RMSE=round(res['D联合_RMSE'].mean(), 3),
    B_up=round(res['B微调较A提升%'].mean(), 2), D_up=round(res['D联合较A提升%'].mean(), 2),
    B_win=int((res['B微调较A提升%'] > 0).sum()), D_win=int((res['D联合较A提升%'] > 0).sum()),
)
print()
print('=' * 78)
print('写报告可直接引用的汇总数字')
print('=' * 78)
for k, v in summ.items():
    print(f'  {k:8s} = {v}')
