from Darcy_Generators import DarcyGenerator, DarcySimParams
import matplotlib.pyplot as plt
import fenics as fe
import numpy as np

# f="6*x[0]+6*x[1]",
# g="pow(x[0],3)+pow(x[1],3)",
# u_expression = ("3*pow(x[0],2)", "3*pow(x[1],2)")
# p_expression = "pow(x[0],3)+pow(x[1],3)"

# def test_darcy_generator():
if __name__ == "__main__":
    hs = [0.5, 0.25, 0.125]
    degree = 1
    u_expression = ("1", "1")
    p_expression = "1-pow(x[0],2)+pow(x[1],2)"
    for h in hs:
        print(f"{h = }")
        test_params = DarcySimParams(
            h=h,
            mesh=fe.UnitSquareMesh(
                round(1 / (h * np.sqrt(2))),
                round(1 / (h * np.sqrt(2))),
            ),
            degree=degree,
            f="0",
            g="0",
        )
        A = np.array([[1, 0], [0, 1]])
        generator = DarcyGenerator(test_params)

        print(f"{generator.mesh.num_cells() = }")

        (u, p) = generator.solve_variational_form(A).split()
        print(f"{len(u.vector().get_local()) = }")
        print(f"{np.max(np.abs(u.vector().get_local())) = }")
        plt.figure()

        fe.plot(u)
        plt.show()
        (u_exact, p_exact) = fe.interpolate(
            fe.Expression(
                (u_expression[0], u_expression[1], p_expression), degree=degree
            ),
            generator.model_space,
        ).split()

        fe.set_log_level(50)
        dx = fe.dx(domain=generator.mesh)
        u_err = np.sqrt(
            fe.assemble(((u[0] - u_exact[0]) ** 2 + (u[1] - u_exact[1]) ** 2) * dx)
        )

        p_err = np.sqrt(fe.assemble((p - p_exact) ** 2 * dx))

        print(f"{u_err = }")
        print(f"{p_err = }")
        # (u_diff, p_diff) = fe.Function(generator.model_space).split()

        # u_diff.vector()[:] = u.vector()[:] - u_exact.vector()[:]
        # p_diff.vector()[:] = p.vector()[:] - p_exact.vector()[:]

        # u_norm = u_diff.vector().norm("l2")
        # p_norm = p_diff.vector().norm("l2")

        # print(f"{u_norm = }")
        # print(f"{p_norm = }")

        # print(f"{len(u_diff.vector()) = }")
        # print(f"{len(p_diff.vector()) = }")
    # assert u_norm < 0.001 and p_norm < 0.001
