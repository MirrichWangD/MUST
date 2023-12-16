from docplex.mp.model import Model


class Ex3:

    @classmethod
    def model(cls):
        n = 5
        c = [[0.0, 24.0, 10.0, 15.0, 12.0],
             [24.0, 0.0, 22.0, 25.0, 20.0],
             [10.0, 22.0, 0.0, 18.0, 30.0],
             [15.0, 25.0, 18.0, 0.0, 34.0],
             [12.0, 20.0, 30.0, 34.0, 0.0]]

        cp = Model()
        x = [[cp.binary_var(name="x_{0}_{1}".format(i + 1, j + 1)) for j in range(n)] for i in range(n)]
        u = [cp.integer_var(name="u_{0}".format(i + 1), lb=1, ub=n) for i in range(n)]

        expr = 0
        for i in range(n):
            for j in range(n):
                if j != i:
                    expr += c[i][j] * x[i][j]
        cp.minimize(expr)

        for j in range(n):
            expr = 0
            for i in range(n):
                if i != j:
                    expr += x[i][j]
            cp.add_constraint(ct=expr == 1.0, ctname="c1_{0}".format(j))

        for i in range(n):
            expr = 0
            for j in range(n):
                if i != j:
                    expr += x[i][j]
            cp.add_constraint(ct=expr == 1.0, ctname="c2_{0}".format(i))

        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    cp.add_constraint(ct=u[i] - u[j] + n * x[i][j] <= n - 1, ctname="c3_{0}_{1}".format(i, j))

        cp.export_as_lp("Ex3.lp")
        solution = cp.solve()
        print()
        print(solution)


if __name__ == "__main__":
    Ex3.model()
