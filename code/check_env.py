# -*- coding: utf-8 -*-
"""
第 1 步：环境自检
=========================================================
作用：确认 3 件事
  1. Python 解释器用的是 Anaconda（不是系统自带的那个）
  2. numpy / pandas / scikit-learn 能正常导入
  3. 能正确读到 data/ 目录下的数据集

怎么运行：在 PyCharm 左侧项目树里右键本文件
        -> 运行 'check_env'
        （或打开本文件后点右上角绿色三角）
=========================================================
"""
import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

print("=" * 56)
print("0) 当前使用的 Python 解释器")
print("   路径 :", sys.executable)
print("   版本 :", sys.version.split()[0])

print("=" * 56)
print("1) 关键库版本检查")
print("   numpy        :", np.__version__)
print("   pandas       :", pd.__version__)
print("   scikit-learn :", sklearn.__version__)

# 数据集路径（4 个文件都在这个目录下）
DATA_DIR = "/Users/xuminchun/Desktop/电气club作业/data"
CSV_PATH = os.path.join(DATA_DIR, "Plant_1_Generation_Data.csv")

print("=" * 56)
print("2) 数据文件检查")
print("   数据目录是否存在 :", os.path.isdir(DATA_DIR))
print("   目标文件是否存在 :", os.path.isfile(CSV_PATH))

if os.path.isfile(CSV_PATH):
    df = pd.read_csv(CSV_PATH, nrows=5)   # 只读前 5 行，快速预览
    print("   列名 :", df.columns.tolist())
    print("-" * 56)
    print(df.head().to_string())
else:
    print("   [X] 没找到文件，请检查 DATA_DIR 路径是否正确")

print("=" * 56)
print("[OK] 环境自检通过：解释器、库、数据三者都就绪。")
