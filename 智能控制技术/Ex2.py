from docplex.mp.model import Model


class Ex2:

    @classmethod
    def model(cls):
        cp = Model()

        ub = [3.0, 1.0, 3.0, 2.0]
        x = cp.integer_var_list(4, name="x", ub=ub)
        cp.maximize(2.0 * x[0] + 4.0 * x[1] + 4.0 * x[2] + 5.0 * x[3])
        cp.add_constraint(ct=1.0 * x[0] + 2.0 * x[1] + 3.0 * x[2] + 4.0 * x[3] <= 5.0, ctname="c1")
        solution = cp.solve()
        print()
        print(solution)


if __name__ == "__main__":
    Ex2.model()
