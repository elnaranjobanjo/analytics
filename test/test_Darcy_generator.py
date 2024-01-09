import matplotlib.pyplot as plt
import fenics as fe
import numpy as np
import sys

sys.path.append("../src/formulations/")
sys.path.append("../src/formulations/PDEs")
sys.path.append("../src/FEM_solvers/")

import formulation as F
import FEM_solver as S
import Darcy as D


def test_dual_convergence(
    hs: list, test_params: F.formulation_params, p_expression: str, u_expression: str
) -> None:
    test_params.formulation = "dual"
    for h in hs:
        test_params.mesh = fe.UnitSquareMesh(
            round(1 / (h * np.sqrt(2))),
            round(1 / (h * np.sqrt(2))),
        )

        print(f"{h = }")
        generator = S.Darcy_FEM_Solver(test_params)
        print(f"{generator.formulation.get_mesh().num_cells() = }")
        (u, p) = generator.solve_variational_form(np.array([[5, 1], [1, 5]])).split()
        print(f"{len(u.vector().get_local()) = }")
        print(f"{np.max(np.abs(u.vector().get_local())) = }")

        # plt.figure()
        # fe.plot(u)
        # plt.show()
        (u_exact, p_exact) = fe.interpolate(
            fe.Expression(
                (u_expression[0], u_expression[1], p_expression),
                degree=test_params.degree,
            ),
            generator.formulation.get_model_space(),
        ).split()

        fe.set_log_level(50)
        dx = fe.dx(domain=generator.formulation.get_mesh())
        u_err = np.sqrt(
            fe.assemble(((u[0] - u_exact[0]) ** 2 + (u[1] - u_exact[1]) ** 2) * dx)
        )
        p_err = np.sqrt(fe.assemble((p - p_exact) ** 2 * dx))

        print(f"{u_err = }")
        print(f"{p_err = }\n")
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


def test_primal_convergence(
    hs: list, test_params: F.formulation_params, p_expression: str
) -> None:
    test_params.formulation = "primal"
    for h in hs:
        test_params.mesh = fe.UnitSquareMesh(
            round(1 / (h * np.sqrt(2))),
            round(1 / (h * np.sqrt(2))),
        )

        print(f"{h = }")

        generator = S.Darcy_FEM_Solver(test_params)
        print(f"{generator.formulation.get_mesh().num_cells() = }")
        p = generator.solve_variational_form(np.array([[5, 1], [1, 5]]))

        p_exact = fe.interpolate(
            fe.Expression(p_expression, degree=test_params.degree),
            generator.formulation.get_model_space(),
        )

        fe.set_log_level(50)
        p_err = np.sqrt(
            fe.assemble(
                (p - p_exact) ** 2 * fe.dx(domain=generator.formulation.get_mesh())
            )
        )

        # p_err = np.sqrt(
        #     fe.assemble((p - p_exact) ** 2 * fe.dx(domain=test_params.mesh))
        # )

        print(f"{p_err = }\n")


if __name__ == "__main__":
    hs = [0.25, 0.125, 0.0625]
    p_expression = "x[0]*(1-x[0])*x[1]*(1-x[1])"
    test_params = F.formulation_params(
        PDE="Darcy_primal",
        degree=3,
        f="-(2*(1-2*x[0])*(1-2*x[1])-10*x[1]*(1-x[1])-10*x[0]*(1-x[0]))",
    )

    print("running primal formulation convergence test\n\n")
    test_primal_convergence(hs, test_params, p_expression)
    test_params.PDE = "Darcy_dual"
    u_expression = (
        "5*x[1]*(1-x[1])*(1-2*x[0])+x[0]*(1-x[0])*(1-2*x[1])",
        "5*x[0]*(1-x[0])*(1-2*x[1])+x[1]*(1-x[1])*(1-2*x[0])",
    )
    print("\nrunning dual formulation convergence test\n\n")
    test_dual_convergence(hs, test_params, p_expression, u_expression)
