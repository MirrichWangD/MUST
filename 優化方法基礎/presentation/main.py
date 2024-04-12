# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : main.py
@ Time        : 2024/04/12 13:57:17
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
此处尝试调用线性求解器来求解TSP问题ulysses16
问题中包含16个城市的x y坐标，数据存储在ulysses16.csv中
+++++++++++++++++++++++++++++++++++
"""

# 导入库
import pulp
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

"""+++++++++++
@@@ Config
++++++++++++++"""

params = {
    "font.family": "Microsoft YaHei",
    "figure.figsize": [3.0, 3.0],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 5,
    "legend.fontsize": "small",
}
mpl.rcParams.update(params)

"""++++++++++++++
Custom function
+++++++++++++++++"""


def distance(pdFrame):
    """从读入的pandas frame中计算每两点之间的距离
    input: pdFrame -- pandas frame，三列：分别为cityIndex, xCoor, yCoor
    output: dict
    """
    return dict(
        ((a, b), np.linalg.norm(pdFrame.values[a - 1] - pdFrame.values[b - 1]))
        for a in pdFrame.index
        for b in pdFrame.index
        if a != b
    )


def visualizePath(pdFrame, edges, save="./figure"):
    """对计算的结果进行可视化
    input: pdFrame -- pandas frame，包含了cityIndex, xCoor, yCoor
           edges -- list, 每个元素是一个表示(From, to)的节点二元组
    output: None
    """

    # 画出所有点
    for i, row in pdFrame.iterrows():
        plt.plot(row.X_COOR, row.Y_COOR, "bo", markersize=2)
        plt.text(row.X_COOR + 0.2, row.Y_COOR + 0.5, "%d" % i, ha="center", va="bottom", fontsize=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{save}/TSP_pos.png")

    # 画出所有点
    for i, row in pdFrame.iterrows():
        plt.plot(row.X_COOR, row.Y_COOR, "bo", markersize=2)
        plt.text(row.X_COOR + 0.2, row.Y_COOR + 0.5, "%d" % i, ha="center", va="bottom", fontsize=5)
    # 画出连线
    for i, j in edges:
        x_i = pdFrame.iloc[i - 1].X_COOR
        y_i = pdFrame.iloc[i - 1].Y_COOR
        x_j = pdFrame.iloc[j - 1].X_COOR
        y_j = pdFrame.iloc[j - 1].Y_COOR
        plt.arrow(
            x_i,
            y_i,
            x_j - x_i,
            y_j - y_i,
            # width=0.2,
            length_includes_head=True,
            head_width=0.3,
            head_length=0.5,
            joinstyle="round",
        )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{save}/TSP_tour.png")


if __name__ == "__main__":
    # 读入数据
    fname = "./data/ulysses16TSP.csv"
    pdFrame = pd.read_csv(fname, index_col=0)
    dist = distance(pdFrame)
    sites = pdFrame.index

    # 创建问题
    model = pulp.LpProblem(fname.split(".")[0], pulp.LpMinimize)

    # 创建变量
    x = pulp.LpVariable.dicts("x", dist, 0, 1, pulp.LpBinary)

    # 设置目标函数
    model += pulp.lpSum([x[(i, j)] * dist[(i, j)] for (i, j) in dist])

    # 施加约束
    # 每个节点只有一条in_edge，一条out_edge
    probSize = len(sites)
    for k in sites:
        model += pulp.lpSum([x[(i, k)] for i in sites if (i, k) in x]) == 1
        model += pulp.lpSum([x[(k, i)] for i in sites if (k, i) in x]) == 1
    # 让每个节点都被经过，避免形成多个subtour闭环
    # 变量u代表在路径上访问一个节点的顺序
    u = pulp.LpVariable.dicts("u", sites, lowBound=1, upBound=probSize - 1, cat=pulp.LpContinuous)
    for i, j in dist:
        if i != 1 and j != 1:
            model += u[i] - u[j] <= probSize * (1 - x[(i, j)]) - 1

    # 求解
    start_time = time.time()
    model.solve()
    print("Time consumed:", time.time() - start_time, "s")
    print("Status:", pulp.LpStatus[model.status])
    print("Minimal route length:", pulp.value(model.objective))

    # 结果输出
    tourVar = [x[(i, j)] for (i, j) in dist]
    print("Paths picked: ")
    for var in tourVar:
        if pulp.value(var) > 0:
            print(var, ":", pulp.value(var))

    print("City Rank:")
    uVar = [u[i] for i in range(2, probSize+1)]
    for var in uVar:
        print(var, ":", pulp.value(var))

    # 可视化
    edges = []
    for i, j in dist:
        if pulp.value(x[(i, j)]) > 0:
            edges.append((i, j))
    visualizePath(pdFrame, edges, "./figure")
