#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 LSTM 的迁移结果与 LightGBM 的结果放在一起比较，并出图。

五个方案：
  A 从零     B 预训练+微调     C 零样本     D 多站联合     E 预训练初始化+训满

产出：
  results/lstm_transfer_comparison.png
  results/lstm_vs_lgbm.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.expanduser('~/Desktop/电气club作业')
RES = os.path.join(BASE, 'results')

for f in ['Heiti SC', 'PingFang SC', 'Songti SC', 'Arial Unicode MS']:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f]
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

lstm = pd.read_csv(os.path.join(RES, 'lstm_transfer_results.csv'), encoding='utf-8-sig')
lgbm = pd.read_csv(os.path.join(RES, 'transfer_loso_results.csv'), encoding='utf-8-sig')

TAGS = ['A从零', 'B微调', 'C零样本', 'D联合', 'E预训练初始化']
LABEL = ['A\n从零', 'B\n预训练+微调', 'C\n零样本', 'D\n多站联合', 'E\n预训练初始化']
COLORS = ['#4C72B0', '#DD8452', '#C44E52', '#55A868', '#8172B3']
NST = len(lstm)

# ---------------- 图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# (a) 各方案平均 MAE
ax = axes[0, 0]
means = [lstm[f'{t}_MAE'].mean() for t in TAGS]
bars = ax.bar(LABEL, means, color=COLORS, edgecolor='black', linewidth=0.6)
for b, v in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, v + max(means) * 0.02, f'{v:.3f}',
            ha='center', fontsize=10)
ax.set_ylabel('平均 MAE（%装机容量）')
ax.set_title('(a) LSTM 五方案平均误差', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(means) * 1.25)

# (b) 逐站改善幅度
ax = axes[0, 1]
x = np.arange(NST)
w = 0.2
for i, t in enumerate(['B微调', 'C零样本', 'D联合', 'E预训练初始化']):
    v = lstm[f'{t}较A提升%'].values
    ax.bar(x + (i - 1.5) * w, v, w, label=t, color=COLORS[i + 1],
           edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([s.replace('station', 'S') for s in lstm['站']], rotation=45)
ax.set_ylabel('相对从零训练的改善（%）')
ax.set_title('(b) 逐站改善幅度（负值为负迁移）', fontsize=11)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_yscale('symlog', linthresh=10)
ax.set_yticks([-200, -100, -50, -20, -10, 0, 10, 20, 50])
ax.set_yticklabels(['-200', '-100', '-50', '-20', '-10', '0', '10', '20', '50'])

# (c) 微调 vs 预训练初始化：B 与 E 的逐站对比
ax = axes[1, 0]
w = 0.38
ax.bar(x - w / 2, lstm['B微调较A提升%'].values, w, label='B 预训练+微调(5轮,小步长)',
       color='#DD8452', edgecolor='black', linewidth=0.5)
ax.bar(x + w / 2, lstm['E预训练初始化较A提升%'].values, w, label='E 预训练初始化+训满',
       color='#8172B3', edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([s.replace('station', 'S') for s in lstm['站']], rotation=45)
ax.set_ylabel('相对从零训练的改善（%）')
ax.set_title('(c) 微调轮数够不够：B 与 E 的差别', fontsize=11)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_yscale('symlog', linthresh=10)
ax.set_yticks([-40, -30, -20, -10, 0, 10, 20, 30])
ax.set_yticklabels(['-40', '-30', '-20', '-10', '0', '10', '20', '30'])

# (d) LSTM vs LightGBM：微调收益对比
ax = axes[1, 1]
w = 0.36
ax.bar(x - w / 2, lstm['B微调较A提升%'].values, w, label='LSTM 微调',
       color='#DD8452', edgecolor='black', linewidth=0.5)
ax.bar(x + w / 2, lgbm['B微调较A提升%'].values, w, label='LightGBM 微调',
       color='#8C8C8C', edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([s.replace('station', 'S') for s in lstm['站']], rotation=45)
ax.set_ylabel('微调相对从零训练的改善（%）')
ax.set_title('(d) 微调收益：LSTM vs LightGBM', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
png = os.path.join(RES, 'lstm_transfer_comparison.png')
plt.savefig(png, dpi=160, bbox_inches='tight')
print('已保存:', png)

# ---------------- 汇总表 ----------------
print('\n' + '=' * 76)
print('LSTM 结果汇总（全天口径，%装机容量）')
print('=' * 76)
tab = pd.DataFrame({
    '方案': TAGS,
    'MAE': [lstm[f'{t}_MAE'].mean() for t in TAGS],
    'RMSE': [lstm[f'{t}_RMSE'].mean() for t in TAGS],
    'MAE白天': [lstm[f'{t}_MAE白天'].mean() for t in TAGS],
    '改善%': ['—'] + [f'{lstm[f"{t}较A提升%"].mean():+.2f}' for t in TAGS[1:]],
    '赢过A的站数': ['—'] + [f'{(lstm[f"{t}较A提升%"] > 0).sum()}/{NST}' for t in TAGS[1:]],
})
print(tab.to_string(index=False))

print('\n' + '=' * 76)
print('同一批数据上，LSTM 与 LightGBM 的迁移收益对比')
print('=' * 76)
cmp = pd.DataFrame({
    '指标': ['A 从零 MAE', 'B 微调 改善%', 'B 赢过 A 的站数',
             'C 零样本 改善%', 'D 联合 改善%', 'D 赢过 A 的站数'],
    'LSTM': [f"{lstm['A从零_MAE'].mean():.3f}",
             f"{lstm['B微调较A提升%'].mean():+.2f}",
             f"{(lstm['B微调较A提升%'] > 0).sum()}/{NST}",
             f"{lstm['C零样本较A提升%'].mean():+.2f}",
             f"{lstm['D联合较A提升%'].mean():+.2f}",
             f"{(lstm['D联合较A提升%'] > 0).sum()}/{NST}"],
    'LightGBM': [f"{lgbm['A从零_MAE'].mean():.3f}",
                 f"{lgbm['B微调较A提升%'].mean():+.2f}",
                 f"{(lgbm['B微调较A提升%'] > 0).sum()}/{NST}",
                 f"{lgbm['C零样本较A提升%'].mean():+.2f}",
                 f"{lgbm['D联合较A提升%'].mean():+.2f}",
                 f"{(lgbm['D联合较A提升%'] > 0).sum()}/{NST}"],
})
print(cmp.to_string(index=False))
cmp.to_csv(os.path.join(RES, 'lstm_vs_lgbm.csv'), index=False, encoding='utf-8-sig')

print('\n' + '=' * 76)
print('关键问题一：LSTM 的微调，是否比树模型的"接着加树"更接近真微调？')
print('=' * 76)
b_lstm = lstm['B微调较A提升%'].mean()
b_lgbm = lgbm['B微调较A提升%'].mean()
print(f'  LSTM     微调改善 = {b_lstm:+.2f}%')
print(f'  LightGBM 微调改善 = {b_lgbm:+.2f}%')
print(f'  差值 = {b_lstm - b_lgbm:+.2f} 个百分点')
if b_lstm > b_lgbm:
    print('  → LSTM 的微调收益更高，支持"可微模型的微调比树模型更彻底"这一解释。')
else:
    print('  → 树模型的"接着加树"并不逊色，说明本任务中限制因素可能不在"能否改老参数"。')

print('\n' + '=' * 76)
print('关键问题二：B 不行是"预训练没用"，还是"微调没训够"？')
print('=' * 76)
b = lstm['B微调较A提升%']
e = lstm['E预训练初始化较A提升%']
print(f'  B（预训练 + 5 轮小步长微调）平均 {b.mean():+.2f}%   赢 {(b > 0).sum()}/{NST} 站')
print(f'  E（预训练初始化 + 15 轮常规训练）平均 {e.mean():+.2f}%   赢 {(e > 0).sum()}/{NST} 站')
print(f'  E - B = {e.mean() - b.mean():+.2f} 个百分点')
if e.mean() > b.mean() + 2:
    print('  → E 明显好于 B：预训练的价值主要在"给了一个好起点"，')
    print('     但必须配合足够的训练轮数才能兑现；少轮数小步长的经典微调反而把它锁在源站附近。')
elif abs(e.mean() - b.mean()) < 2:
    print('  → B 与 E 接近：微调轮数不是瓶颈，预训练带来的初始化优势本身就有限。')
else:
    print('  → B 好于 E：少轮数小步长更合适，训得多反而把预训练知识忘掉了（灾难性遗忘）。')

# 负迁移案例
neg = lstm[lstm['B微调较A提升%'] < 0]
print(f'\nLSTM 负迁移站点数: {len(neg)} / {NST}')
if len(neg):
    print(neg[['站', 'A从零_MAE', 'B微调_MAE', 'B微调较A提升%',
               'D联合较A提升%', 'E预训练初始化较A提升%']].to_string(index=False))

# 相关性：基线越弱，迁移收益越大？
try:
    from scipy.stats import pearsonr
    r, p = pearsonr(lstm['A从零_MAE'], lstm['E预训练初始化较A提升%'])
    print(f'\n基线 MAE 与 E 改善幅度的相关性: r = {r:.3f}, p = {p:.3f}'
          f'{"（未达显著，样本量不足）" if p >= 0.05 else "（显著）"}')
except Exception as ex:
    print('\n相关性检验跳过:', ex)
