# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : graphical_solution.py
@ Time        : 2024/04/29 18:14:34
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
...
+++++++++++++++++++++++++++++++++++
"""


import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist


def f(x):
    return 6 - x


def obj(x, value):
    return value / 3 - 2 / 3 * x


# ------------- #
# Config
# ------------- #
plt.clf()
# fig, ax = plt.subplots()
fig = plt.figure()
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
ax = axisartist.Subplot(fig, 111)
fig.add_axes(ax)
# 去除右边、上边边框线
ax.axis[:].set_visible(False)
# ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0, 0)
# 给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("-|>", size=1.0)
# 添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["y"].set_axisline_style("-|>", size=1.0)
# 设置x、y轴上刻度显示方向
# ax.axis["x"].set_axis_direction("top")
ax.axis["y"].set_axis_direction("left")

# 轴线刻度长度
ax.tick_params(which='minor', length=18)

# ------------- #
# Visual
# ------------- #

# 设置坐标轴刻度
x_min, x_max = 0, 8
y_min, y_max = 0, 7
ax.set_ylim(y_min, y_max)
ax.set_xlim(x_min, x_max)
ax.set_xticks(range(x_min, x_max))
ax.set_yticks(range(y_min, y_max))
plt.text(8, -.3, r"$x_1$")
plt.text(-.3, 7, r"$x_2$")

# 绘制线
plt.vlines(4, 0, 6, colors="k")  # x_1 = 4
plt.hlines(3, 0, 5, colors="k")  # 2x_2 = 12

x1 = np.arange(0, 7)
x2 = f(x1)

plt.plot(x1, x2, color="k")

# 多边形填充颜色
polygon = [[0, 3], [3, 3], [4, 2], [4, 0], [0, 0], [0, 3]]
plt.fill([0, 3, 4, 4, 0, 0], [3, 3, 2, 0, 0, 3])

plt.text(0.5, 5.7, r"$2x_1+2x_2=12$")
plt.text(4.2, 5.5, r"$4x_1=16$")
plt.text(5.2, 2.8, r"$5x_2=15$")
plt.text(1, 1.5, "Feasible\n region")

# 绘制目标函数等高线
# x1 = np.arange(-1, 7)
# values = np.arange(6, 13, 2)
# x2s = [obj(x1, i) for i in values]
#
# for i, x2 in enumerate(x2s):
#     plt.text(1.7, x2[3]-0.1, f"$Z={values[i]}$", rotation=-32, ha="center", va="center")
#     plt.plot(x1, x2, color="k")


plt.show()
