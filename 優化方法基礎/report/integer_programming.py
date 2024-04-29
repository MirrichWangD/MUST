import pulp

# 创建整数规划问题
problem = pulp.LpProblem("Integer_Problem", pulp.LpMaximize)

# 定义决策变量
x1 = pulp.LpVariable("x1", lowBound=0, upBound=3, cat="Integer")
x2 = pulp.LpVariable("x2", lowBound=0, upBound=1, cat="Integer")
x3 = pulp.LpVariable("x3", lowBound=0, upBound=3, cat="Integer")
x4 = pulp.LpVariable("x4", lowBound=0, upBound=2, cat="Integer")

# 添加目标函数
problem += 2 * x1 + 4 * x2 + 4 * x3 + 5 * x4, "Objective Function"

# 添加约束条件
problem += x1 + 2 * x2 + 3 * x3 + 4 * x4 <= 5, "Constraint_1"

# 求解问题
problem.solve()

# 打印结果
print("Optimal Value of Z:", pulp.value(problem.objective))
print("Optimal Values of x1, x2, x3, x4:", x1.varValue, x2.varValue, x3.varValue, x4.varValue)
