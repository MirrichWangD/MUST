# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : demo.py
@ Time        : 2023/11/14 13:32:32
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
True Latitude Longitude: (22.13807371546038, 113.5380982705002)
Only use latitude and longitude to complete task.
WGS84 to ECEF is using pyproj module
+++++++++++++++++++++++++++++++++++
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

warnings.filterwarnings("ignore")

"""=========================
@@@ Initialize Transformer
========================="""

wgs2utm = Transformer.from_crs("epsg:4326", "epsg:32649")
utm2wgs = Transformer.from_crs("epsg:32649", "epsg:4326")

"""==============================
@@@ Data Loading and Preprossing
=============================="""

data = pd.read_csv("./data/GPSdata1206.csv")
data = data.drop_duplicates()  # drop duplicates

latitude = data["latitude"].to_numpy()
longitude = data["longitude"].to_numpy()
altitude = data["altitude"].to_numpy()
accuracy = data["accuracy"].to_numpy()

# wgs84 to utm (x, y)
x_lst, y_lst = [], []
for lat, lon in zip(latitude, longitude):
    x, y = wgs2utm.transform(lat, lon)
    x_lst.append(x)
    y_lst.append(y)

# convert to array
xx = np.array(x_lst)
yy = np.array(y_lst)
d = accuracy


# Min-Max standardization
x_min, x_max = xx.min(), xx.max()
y_min, y_max = yy.min(), yy.max()
d_min, d_max = d.min(), d.max()

x = (xx - x_min) / (x_max - x_min)
y = (yy - y_min) / (y_max - y_min)
d = (d - d_min) / (d_max - d_min)


"""======================
@@@ Least Squares Method
======================"""


A = 2 * np.concatenate([[x[1:] - x[:-1]], [y[1:] - y[:-1]]], axis=0).T

B = d[:-1] ** 2 - d[1:] ** 2 - (x[:-1] ** 2 + y[:-1] ** 2) + (x[1:] ** 2 + y[1:] ** 2)

# solution
X = np.linalg.inv(A.T @ A) @ (A.T @ B)

# restore ture x, y, z
x_pred = X[0] * (x_max - x_min) + x_min
y_pred = X[1] * (y_max - y_min) + y_min

# restore latitude, longitude
lat_pred, lon_pred = utm2wgs.transform(x_pred, y_pred)

print("x_pred: %f, y_pred: %f" % (x_pred, y_pred))
print("lat_pred: %f, lon_pred: %f" % (lat_pred, lon_pred))

plt.plot(longitude, latitude, "b*")
plt.plot(lon_pred, lat_pred, "r.")
plt.show()
