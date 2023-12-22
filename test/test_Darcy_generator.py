import matplotlib.pyplot as plt
import fenics as fe
import numpy as np
import sys

sys.path.append("../src/generators/")

import Darcy_generator as Dg


from dolfin import *


def solver_online(A, mesh, degree, f):
    # Create mesh and define function space

    V = FunctionSpace(mesh, "Lagrange", degree)

    def boundary(x):
        return (
            x[0] < fe.DOLFIN_EPS
            or x[0] > 1.0 - fe.DOLFIN_EPS
            or x[1] < fe.DOLFIN_EPS
            or x[1] > 1.0 - fe.DOLFIN_EPS
        )

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    df = Expression(f, degree=degree)
    a = inner(fe.Constant(A) * grad(u), grad(v)) * dx
    L = df * v * dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)
    return u, V


def test_dual_convergence(
    hs: list, test_params: Dg.DarcySimParams, p_expression: str, u_expression: str
) -> None:
    test_params.formulation = "dual"
    for h in hs:
        test_params.mesh = fe.UnitSquareMesh(
            round(1 / (h * np.sqrt(2))),
            round(1 / (h * np.sqrt(2))),
        )

        print(f"{h = }")
        generator = Dg.Darcy_FEM_Solver(test_params)
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
    hs: list, test_params: Dg.DarcySimParams, p_expression: str
) -> None:
    test_params.formulation = "primal"
    for h in hs:
        test_params.mesh = fe.UnitSquareMesh(
            round(1 / (h * np.sqrt(2))),
            round(1 / (h * np.sqrt(2))),
        )

        print(f"{h = }")

        # generator = Dg.Darcy_FEM_Solver(test_params)
        # print(f"{generator.formulation.get_mesh().num_cells() = }")
        # p = generator.solve_variational_form(np.array([[5, 1], [1, 5]]))
        p, V = solver_online(
            np.array([[5, 1], [1, 5]]),
            test_params.mesh,
            test_params.degree,
            test_params.f,
        )

        # plt.figure()
        # fe.plot(p)
        # plt.show()

        # p_exact = fe.interpolate(
        #     fe.Expression(p_expression, degree=test_params.degree),
        #     generator.formulation.get_model_space(),
        # )
        p_exact = fe.interpolate(
            fe.Expression(p_expression, degree=test_params.degree),
            V,
        )
        fe.set_log_level(50)
        # p_err = np.sqrt(
        #     fe.assemble(
        #         (p - p_exact) ** 2 * fe.dx(domain=generator.formulation.get_mesh())
        #     )
        # )

        p_err = np.sqrt(
            fe.assemble((p - p_exact) ** 2 * fe.dx(domain=test_params.mesh))
        )

        print(f"{p_err = }\n")


if __name__ == "__main__":
    hs = [0.25, 0.125, 0.0625]
    p_expression = "x[0]*(1-x[0])*x[1]*(1-x[1])"
    test_params = Dg.DarcySimParams(
        degree=3,
        f="2*(1-2*x[0])*(1-2*x[1])-10*x[1]*(1-x[1])-10*x[0]*(1-x[0])",
    )

    print("running primal formulation convergence test\n\n")
    test_primal_convergence(hs, test_params, p_expression)
    u_expression = (
        "5*x[1]*(1-x[1])*(1-2*x[0])+x[0]*(1-x[0])*(1-2*x[1])",
        "5*x[0]*(1-x[0])*(1-2*x[1])+x[1]*(1-x[1])*(1-2*x[0])",
    )
    print("\nrunning dual formulation convergence test\n\n")
    # test_dual_convergence(hs, test_params, p_expression, u_expression)
