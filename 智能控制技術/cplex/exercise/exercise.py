# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : exercise.py
@ Time        : 2023/11/17 23:23:30
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
Exercise:
    Given 3 classes of tasks T1, T2, T3 and 2 classes of machines M1, M2. Now 5 T1, 3 T2,
    and 2 T3 are scheduled on 1 M1 and 2 M2. A machine only tackles a task at a time.
    The time taken by different machines for different tasks is showed as follows.
    The problem is to minimize the makespan such that 10 tasks are completed.

------------------------------
|    |   T1  |   T2  |   t3  |
| M1 |   10  |   12  |   18  |
| M2 |   15  |   8   |   20  |
------------------------------

Solution:

------------------------------
|    |   T1  |   T2  |   t3  |
| M1 |   x1  |   x2  |   x3  |
| M2 |   x4  |   x5  |   x6  |
------------------------------

min z = 10 * x1 + 12 * x2 + 18 * x3 + 15 * x4 + 8 * x5 + 20 * x6

s.t.    x1 + x2 + x3 + x4 + x5 + x6 >= 10
        x1 + 2 * x4 >= 5,
        x2 + 2 * x5 >= 3,
        x3 + 2 * x6 >= 2,
        0 <= x1 <= 5,
        0 <= x2 <= 3,
        0 <= x3 <= 2,
        0 <= x4 <= 5,
        0 <= x5 <= 3.
        0 <= x6 <= 2,

Result:
    min z = 76
    ------------------------------
    |    |   T1  |   T2  |   t3  |
    | M1 |   1   |   0   |   0   |
    | M2 |   2   |   2   |   1   |
    ------------------------------

+++++++++++++++++++++++++++++++++++
"""

from docplex.mp.model import Model

# initialize model
cp = Model()

# initialize variables
x1 = cp.integer_var(ub=5, name="x1")
x2 = cp.integer_var(ub=3, name="x2")
x3 = cp.integer_var(ub=2, name="x3")
x4 = cp.integer_var(ub=5, name="x4")
x5 = cp.integer_var(ub=3, name="x5")
x6 = cp.integer_var(ub=2, name="x6")

# defind minimize object function
cp.minimize(10 * x1 + 12 * x2 + 18 * x3 + 15 * x4 + 8 * x5 + 20 * x6)

# defind constraints
cp.add_constraint(x1 + x2 + x3 + 2 * (x4 + x5 + x6) >= 10)
cp.add_constraint(x1 + 2 * x4 >= 5)
cp.add_constraint(x2 + 2 * x5 >= 3)
cp.add_constraint(x3 + 2 * x6 >= 2)

# solve the problem
solution = cp.solve()

# print result
if solution:
    print(solution)
else:
    print("fail")
