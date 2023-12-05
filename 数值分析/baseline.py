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
from pyproj import Transformer


"""=========================
@@@ Initialize Transformer
========================="""

wgs2utm = Transformer.from_crs("epsg:4326", "epsg:32649")
utm2wgs = Transformer.from_crs("epsg:32649", "epsg:4326")


"""==============================
@@@ Data Loading and Preprossing
=============================="""

data = pd.read_csv("GPSdata_new.csv")
data = data.drop_duplicates()  # drop duplicates

latitude = data["latitude"].to_numpy()
longitude = data["longitude"].to_numpy()
accuracy = data["accuracy"].to_numpy()


# wgs84 to utm (x, y)
x_lst, y_lst = [], []
for lat, lon in zip(latitude, longitude):
    x, y = wgs2utm.transform(lat, lon)
    x_lst.append(x)
    y_lst.append(y)

# convert to array
x = np.array(x_lst)
y = np.array(y_lst)
d = accuracy


# Min-Max standardization
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
d_min, d_max = d.min(), d.max()

x = (x - x_min) / (x_max - x_min)
y = (y - y_min) / (y_max - y_min)
d = (d - d_min) / (d_max - d_min)


"""======================
@@@ Least Squares Method
======================"""


A = np.concatenate([[x[1:] - x[:-1]], [y[1:] - y[:-1]]], axis=0).T

B = d[1:] ** 2 - d[:-1] ** 2 - (x[1:] ** 2 + y[:-1] ** 2) + (x[1:] ** 2 + x[:-1] ** 2)

# solution
X = np.linalg.inv(A.T @ A) @ (A.T @ B)

# restore ture x, y
x_pred = X[0] * (x_max - x_min) + x_min
y_pred = X[1] * (y_max - y_min) + y_min

# restore latitude, longitude
lat_pred, lon_pred = utm2wgs.transform(x_pred, y_pred)

print("x_pred: %f, y_pred: %f" % (x_pred, y_pred))
print("lat_pred: %f, lon_pred: %f" % (lat_pred, lon_pred))
