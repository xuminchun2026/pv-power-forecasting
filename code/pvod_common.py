# -*- coding: utf-8 -*-
"""
pvod_common.py
==============
PVOD 实验的公共工具：数据加载、清洗、时间特征、安全滞后。

【为什么需要"安全滞后"】
  原始数据在若干站点上存在天数级的断点（例如 station08 有一处跳了 3 天）。
  如果直接 shift(1)，断点后第一条记录取到的"上一刻"其实是三天前同一时刻的值，
  滞后特征就完全失真了。所以本模块在生成滞后量时会核对时间间隔，
  间隔不等于 k×15 分钟的一律置为缺失，后续由 dropna 整行剔除。
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "PVOD_hebei")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")

STATIONS = [f"station{i:02d}" for i in range(10)]
RANDOM_STATE = 42
TRAIN_RATIO = 0.7

GHI = "lmd_totalirrad"
NWP_GHI = "nwp_globalirrad"
STEP = pd.Timedelta("15min")

LMD_FEATS = ["lmd_totalirrad", "lmd_diffuseirrad", "lmd_temperature",
             "lmd_pressure", "lmd_windspeed", "lmd_winddirection"]
NWP_FEATS = ["nwp_globalirrad", "nwp_directirrad", "nwp_temperature",
             "nwp_humidity", "nwp_windspeed", "nwp_winddirection", "nwp_pressure"]
TIME_FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

NAME_CN = {
    "lmd_totalirrad": "总辐射 GHI",
    "lmd_diffuseirrad": "散射辐射 DHI",
    "lmd_temperature": "气温",
    "lmd_pressure": "气压",
    "lmd_windspeed": "风速",
    "lmd_winddirection": "风向",
    "nwp_globalirrad": "NWP总辐射",
    "nwp_directirrad": "NWP直接辐射",
    "nwp_temperature": "NWP气温",
    "nwp_humidity": "NWP相对湿度",
    "nwp_windspeed": "NWP风速",
    "nwp_winddirection": "NWP风向",
    "nwp_pressure": "NWP气压",
    "GHI_lag1": "15分钟前辐射",
    "GHI_lag2": "30分钟前辐射",
    "GHI_lag3": "45分钟前辐射",
    "GHI_lag6": "1.5小时前辐射",
    "GHI_lag12": "3小时前辐射",
    "P_lag1": "15分钟前功率",
    "T_lag1": "15分钟前气温",
    "T_lag6": "1.5小时前气温",
    "hour_sin": "时刻(正弦)",
    "hour_cos": "时刻(余弦)",
    "doy_sin": "年内日序(正弦)",
    "doy_cos": "年内日序(余弦)",
}


def capacity_mw(sid):
    m = pd.read_csv(os.path.join(DATA, "metadata.csv")).set_index("Station_ID")
    return float(m.loc[sid, "Capacity"]) / 1000.0


UTC_OFFSET_H = 8   # PVOD 的时间戳是 UTC，+8 才是北京时间


def load_station(sid, clean_night=True):
    """读取单站数据，做时区转换、排序、去重、夜间异常剔除与归一化。

    ⚠️ 时区：原始 date_time 是 UTC。实测证据——按时钟小时统计总辐射，
    峰值出现在 04:00 而不是 12:00（station01 在 117.46°E，当地正午约
    12:07 北京时间＝04:07 UTC）。若不转换，太阳位置、晴空指数、日内曲线全都会错。
    """
    cap = capacity_mw(sid)
    d = pd.read_csv(os.path.join(DATA, f"{sid}.csv"))
    d["date_time"] = pd.to_datetime(d["date_time"]) + pd.Timedelta(hours=UTC_OFFSET_H)
    d = (d.sort_values("date_time")
           .drop_duplicates(subset="date_time", keep="first")
           .reset_index(drop=True))
    if clean_night:
        # 总辐射为 0 却仍有出力 → 夜间异常记录
        d = d[~((d[GHI] <= 0) & (d["power"] > 0.01))].reset_index(drop=True)
    d["p_norm"] = d["power"] / cap
    d.attrs["cap_mw"] = cap
    return d


def add_time_features(d):
    """把「几点 / 一年第几天」映射到圆周上的两个坐标。
    不然 23 点和 0 点在数字上差 23，模型会以为它们离得很远。"""
    t = d["date_time"]
    frac = t.dt.hour + t.dt.minute / 60.0
    d["hour_sin"] = np.sin(2 * np.pi * frac / 24)
    d["hour_cos"] = np.cos(2 * np.pi * frac / 24)
    d["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365)
    d["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365)
    return d


def safe_lag(d, col, k, name=None):
    """滞后 k 步，但只在上一条记录确实是 k×15 分钟之前时才取值。"""
    name = name or f"{col}_lag{k}"
    gap = d["date_time"] - d["date_time"].shift(k)
    v = d[col].shift(k)
    v[gap != k * STEP] = np.nan
    d[name] = v
    return d


def build_b(d):
    """设定乙：纯历史时序特征。全部用 safe_lag，断点处为缺失。"""
    d = add_time_features(d.copy())
    for k in [1, 2, 3, 6, 12]:
        safe_lag(d, GHI, k, f"GHI_lag{k}")
    safe_lag(d, "p_norm", 1, "P_lag1")
    safe_lag(d, "lmd_temperature", 1, "T_lag1")
    safe_lag(d, "lmd_temperature", 6, "T_lag6")
    feats = ["GHI_lag1", "GHI_lag2", "GHI_lag3", "GHI_lag6", "GHI_lag12",
             "P_lag1", "T_lag1", "T_lag6"] + TIME_FEATS
    return d, feats


def feats_a_lmd():
    return LMD_FEATS + TIME_FEATS


def feats_a_nwp():
    return NWP_FEATS + TIME_FEATS


def split_by_time(d):
    """按时间先后切前 70% 训练、后 30% 测试，绝不打乱（打乱=数据泄露）。"""
    cut = int(len(d) * TRAIN_RATIO)
    return d.iloc[:cut], d.iloc[cut:]


def fit_coef(tr, ghi_col):
    """最小二乘估无截距系数 k：p_norm ≈ k × 辐射"""
    x, y = tr[ghi_col].values, tr["p_norm"].values
    return float(np.sum(x * y) / np.sum(x ** 2))
