# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : demo_new.py
@ Time        : 2023/12/07 16:45:47
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
True LLA: (22.13484127150639, 113.5409844704088)
Using latitude, longitude, altitude to complete task.
according to the Algorithm principle of WGS84 -> ECEF, implemented using numpy.
+++++++++++++++++++++++++++++++++++
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.figure(figsize=(8, 6))

"""===========
@@@ Config
==========="""

# the latitude and longitude of ture position
true_lla = (22.13484127150639, 113.5409844704088)
# WGS84 parameters
a = 6378137.0  # Ellipsoid major axis
b = 6356752.314245  # minor semi-axis of ellipsoid
f_inv = 298.257223565  # The reciprocal of WGS84 ellipsoid oblateness: 1/f_inv = (a-b)/a

"""=========================
@@@ Customize function
========================="""


def lla2xyz(latitude, longitude, altitude):
    """
    (Latitude, Longitude, Altitude) -> (x, y, z) in WGS84

    Args:
        latitude (np.array): latitude array
        longitude (np.array): longitude array
        altitude (np.array): altitude array

    Returns:
        np.array: x, y, z
    """
    # 经纬度的余弦值
    cosLat = np.cos(latitude * np.pi / 180)
    sinLat = np.sin(latitude * np.pi / 180)
    cosLon = np.cos(longitude * np.pi / 180)
    sinLon = np.sin(longitude * np.pi / 180)

    N = a / np.sqrt(1 - 1 / f_inv * (2 - 1 / f_inv) * sinLat**2)

    # get the (x, y, z)
    X = (N + altitude) * cosLat * cosLon
    Y = (N + altitude) * cosLat * sinLon
    Z = (N * (1 - 1 / f_inv) ** 2 + altitude) * sinLat

    return X, Y, Z


def xyz2lla(X, Y, Z):
    """
    (x, y, z) -> (Latitude, Longitude, Altitude) in WGS84

    Args:
        X (np.array): x array or float
        Y (np.array): y array or float
        Z (np.array): z array or float

    Returns:
        np.array: latitude, longitude, altitude
    """
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


def get_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the straight line distance between the latitude and longitude of two points

    Args:
        lat1 (float): the latitude of point A
        lon1 (float): the longitude of point A
        lat2 (float): the latitude of point B
        lon2 (float): the longitude of point B

    Returns:
        float: the distance of point A and B
    """
    # GetDistanceInGeographyCoordinate, return two point distance
    radius_lat1 = lat1 * np.pi / 180
    radius_lat2 = lat2 * np.pi / 180
    radius_lon1 = lon1 * np.pi / 180
    radius_lon2 = lon2 * np.pi / 180
    a1 = radius_lat1 - radius_lat2
    b1 = radius_lon1 - radius_lon2
    distance = 2 * np.arcsin(
        np.sqrt(pow(np.sin(a1 / 2.0), 2) + np.cos(radius_lat1) * np.cos(radius_lat2) * pow(np.sin(b1 / 2.0), 2))
    )

    distance = distance * a
    distance = distance - (distance * 0.0011194)
    return distance


"""==============================
@@@ Data Loading and Preprossing
=============================="""

data = pd.read_csv("./data/GPSdata1206.csv")  # load csv format file
data = data.drop_duplicates()  # drop duplicates

# convert pd.Series to numpy.array
latitude = data["latitude"].to_numpy()
longitude = data["longitude"].to_numpy()
altitude = data["altitude"].to_numpy()
accuracy = data["accuracy"].to_numpy()

# convert (latitude, longitude, altitude) to (x, y, z)
xx, yy, zz = lla2xyz(latitude, longitude, altitude)
# lat, lon, alt = XYZ_to_LLA(xx, yy, zz)

# use accuracy as distance
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
# x = xx
# y = yy
# z = zz

"""======================
@@@ Least Squares Method
======================"""

# initialize A matrix
A = 2 * np.concatenate([[x[1:] - x[:-1]], [y[1:] - y[:-1]], [z[1:] - z[:-1]]], axis=0).T
# initialize B maxtix
B = d[:-1] ** 2 - d[1:] ** 2 - (x[:-1] ** 2 + y[:-1] ** 2 + z[:-1] ** 2) + (x[1:] ** 2 + y[1:] ** 2 + z[1:] ** 2)

# solution
X = np.linalg.inv(A.T @ A) @ (A.T @ B)

# restore ture x, y, z
x_pred = X[0] * (x_max - x_min) + x_min
y_pred = X[1] * (y_max - y_min) + y_min
z_pred = X[2] * (z_max - z_min) + z_min
# x_pred = X[0]
# y_pred = X[1]
# z_pred = X[2]

# restore (latitude, longitude, altitude)
lat_pred, lon_pred, alt_pred = xyz2lla(x_pred, y_pred, z_pred)

# print result
print("x_pred: %f, y_pred: %f, z_pred: %f" % (x_pred, y_pred, z_pred))
print("lat_pred: %f, lon_pred: %f, alt_pred: %f" % (lat_pred, lon_pred, alt_pred))
print("The distance of true position: %fm" % (get_distance(true_lla[0], true_lla[1], lat_pred, lon_pred)))


"""=================
@@@ Visualize
================="""

plt.plot(longitude, latitude, "b*", label="data point")
plt.plot(lon_pred, lat_pred, "r.", label="Solution point")
plt.plot(true_lla[1], true_lla[0], "y.", label="true point")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("LLA Visual Figure")
plt.legend()
plt.show()
