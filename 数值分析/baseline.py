# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : solution.py
@ Time        : 2023/11/30 13:32:32
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
...
+++++++++++++++++++++++++++++++++++
"""

import numpy as np
import pandas as pd


data = pd.read_csv("GPSdata_new.csv")
data = data.drop_duplicates()  # 去重
x = data["latitude"].to_numpy()
y = data["longitude"].to_numpy()
d = data["accuracy"].to_numpy()

# 标准化
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
d_min, d_max = d.min(), d.max()

x = (x - x_min) / (x_max - x_min)
y = (y - y_min) / (y_max - y_min)
d = (d - d_min) / (d_max - d_min)


A = np.concatenate([[x[1:] - x[:-1]], [y[1:] - y[:-1]]], axis=0).T

B = d[1:] ** 2 - d[:-1] ** 2 - (x[1:] ** 2 + y[:-1] ** 2) + (x[1:] ** 2 + x[:-1] ** 2)

# 求解坐标
X = np.linalg.inv(A.T @ A) @ (A.T @ B)

# 获取真实经纬度
x_pred = X[0] * (x_max - x_min) + x_min
y_pred = X[1] * (y_max - y_min) + y_min
print(x_pred, y_pred)
