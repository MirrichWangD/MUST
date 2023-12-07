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

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer


"""=========================
@@@ Config
========================="""


a = 6378137.0
f = 1 / 298.257223565


"""=========================
@@@ Initialize Transformer
========================="""

wgs2utm = Transformer.from_crs("epsg:4326", "epsg:32649")
utm2wgs = Transformer.from_crs("epsg:32649", "epsg:4326")


def LLA_to_XYZ(latitude, longitude, altitude):
    """
    (Latitude, Longitude, Altitude) -> (x, y, z) in WGS84

    Args:
        latitude (np.array): latitude array
        longitude (np.array): longitude array
        altitude (np.array): altitude

    Returns:
        np.array: x, y, z
    """
    # 经纬度的余弦值
    cosLat = np.cos(latitude * np.pi / 180)
    sinLat = np.sin(latitude * np.pi / 180)
    cosLon = np.cos(longitude * np.pi / 180)
    sinLon = np.sin(longitude * np.pi / 180)

    # WGS84坐标系的参数
    rad = 6378137.0  # 地球赤道平均半径（椭球长半轴：a）
    f = 1.0 / 298.257223565  # WGS84椭球扁率 :f = (a-b)/a
    C = 1.0 / np.sqrt(cosLat * cosLat + (1 - f) * (1 - f) * sinLat * sinLat)
    S = (1 - f) * (1 - f) * C
    h = altitude

    # 计算XYZ坐标
    X = (rad * C + h) * cosLat * cosLon
    Y = (rad * C + h) * cosLat * sinLon
    Z = (rad * S + h) * sinLat

    return X, Y, Z


def XYZ_to_LLA(X, Y, Z):
    """
    (x, y, z) -> (Latitude, Longitude, Altitude) in WGS84

    Args:
        X (np.array): x array or float
        Y (np.array): y array or float
        Z (np.array): z array or float

    Returns:
        np.array: latitude, longitude, altitude
    """
    # WGS84坐标系的参数
    a = 6378137.0  # 椭球长半轴
    b = 6356752.314245  # 椭球短半轴
    ea = np.sqrt((a**2 - b**2) / a**2)
    eb = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Z * a, p * b)

    # 计算经纬度及海拔
    longitude = np.arctan2(Y, X)
    latitude = np.arctan2(Z + eb**2 * b * np.sin(theta) ** 3, p - ea**2 * a * np.cos(theta) ** 3)
    N = a / np.sqrt(1 - ea**2 * np.sin(latitude) ** 2)
    altitude = p / np.cos(latitude) - N

    return np.degrees(latitude), np.degrees(longitude), altitude


"""==============================
@@@ Data Loading and Preprossing
=============================="""

data = pd.read_csv("GPSdata_new.csv")
data = data.drop_duplicates()  # drop duplicates

latitude = data["latitude"].to_numpy()
longitude = data["longitude"].to_numpy()
altitude = data["altitude"].to_numpy()
accuracy = data["accuracy"].to_numpy()


xx, yy, zz = LLA_to_XYZ(latitude, longitude, altitude)
lat, lon, alt = XYZ_to_LLA(xx, yy, zz)

# # wgs84 to utm (x, y)
# x_lst, y_lst = [], []
# for lat, lon in zip(latitude, longitude):
#     x, y = wgs2utm.transform(lat, lon)
#     x_lst.append(x)
#     y_lst.append(y)

# convert to array
# x = np.array(x_lst)
# y = np.array(y_lst)
d = accuracy


# Min-Max standardization
x_min, x_max = xx.min(), xx.max()
y_min, y_max = yy.min(), yy.max()
z_min, z_max = zz.min(), zz.max()
d_min, d_max = d.min(), d.max()

x = (xx - x_min) / (x_max - x_min)
y = (yy - y_min) / (y_max - y_min)
z = (zz - z_min) / (z_max - z_min)
d = (d - d_min) / (d_max - d_min)

# Z-score standardization
# x_mean, x_std = np.mean(xx), np.std(xx)
# y_mean, y_std = np.mean(yy), np.std(yy)
# z_mean, z_std = np.mean(zz), np.std(zz)
# d_mean, d_std = np.mean(d), np.std(d)

# x = (xx - x_mean) / x_std
# y = (yy - y_mean) / y_std
# z = (zz - z_mean) / z_std
# d = (d - d_mean) / d_std

plt.plot(longitude, latitude, "b*")

"""======================
@@@ Least Squares Method
======================"""


A = 2 * np.concatenate([[x[1:] - x[:-1]], [y[1:] - y[:-1]], [z[1:] - z[:-1]]], axis=0).T

B = d[:-1] ** 2 - d[1:] ** 2 - (x[:-1] ** 2 + y[:-1] ** 2 + z[:-1] ** 2) + (x[1:] ** 2 + y[1:] ** 2 + z[1:] ** 2)

# solution
X = np.linalg.inv(A.T @ A) @ (A.T @ B)

# restore ture x, y, z
x_pred = X[0] * (x_max - x_min) + x_min
y_pred = X[1] * (y_max - y_min) + y_min
z_pred = X[2] * (z_max - z_min) + z_min
# x_pred = X[0] * x_std + x_mean
# y_pred = X[1] * y_std + y_mean
# z_pred = X[2] * z_std + z_mean

# restore latitude, longitude
# lat_pred, lon_pred = utm2wgs.transform(x_pred, y_pred)
lat_pred, lon_pred, alt_pred = XYZ_to_LLA(x_pred, y_pred, z_pred)

print("x_pred: %f, y_pred: %f" % (x_pred, y_pred))
print("lat_pred: %f, lon_pred: %f" % (lat_pred, lon_pred))

# plt.plot(lon_pred, lat_pred, "r.")
plt.show()
