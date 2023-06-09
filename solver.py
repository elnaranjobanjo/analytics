from typing import Tuple
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


# Solves:
#       div u = f
#    A grad p = u
#           p = g  b.c.
# Generates data for the sampling of the mapping A -> (u,p)


@dataclass
class DarcySimParams:
    h: float = 0.1
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    g: str = "0"
    f: str = "1"


class DarcyGenerator:
    mesh: fe.Mesh
    degree: int
    g: fe.Expression
    f: fe.Expression
    model_space: fe.FunctionSpace

    def __init__(self, params: DarcySimParams):
        self.mesh = params.mesh
        self.degree = params.degree
        self.g = fe.Expression(params.g, degree=params.degree)
        self.f = fe.Expression(params.f, degree=params.degree - 1)
        self.model_space = fe.FunctionSpace(
            self.mesh,
            fe.FiniteElement("BDM", self.mesh.ufl_cell(), self.degree)
            * fe.FiniteElement("DG", self.mesh.ufl_cell(), self.degree - 1),
        )

    def compute_solution(
        self, A: np.array, supress_fe_log: bool = True
    ) -> tuple[np.array, np.array]:
        (u, p) = self.solve_variational_form(A, supress_fe_log=supress_fe_log)
        return (u.vector().get_local(), p.vector().get_local())

    def solve_variational_form(
        self,
        A: np.array,
        supress_fe_log: bool = True,
    ) -> tuple[fe.Function, fe.Function]:
        (u, p) = fe.TrialFunctions(self.model_space)
        (v, q) = fe.TestFunctions(self.model_space)
        Ainv = np.linalg.inv(A)
        a = (
            fe.dot(
                fe.Constant(((Ainv[0, 0], Ainv[0, 1]), (Ainv[1, 0], Ainv[1, 1]))) * u,
                v,
            )
            + fe.div(v) * p
            + fe.div(u) * q
        ) * fe.dx
        L = -self.f * q * fe.dx

        if supress_fe_log:
            fe.set_log_level(50)

        sol = fe.Function(self.model_space)
        fe.solve(
            a == L, sol, fe.DirichletBC(self.model_space.sub(1), self.g, "on_boundary")
        )

        return sol.split()


if __name__ == "__main__":
    h = 0.1
    test_params = DarcySimParams(
        h=h,
        mesh=fe.UnitSquareMesh(
            round(1 / (h * np.sqrt(2))),
            round(1 / (h * np.sqrt(2))),
        ),
        degree=3,
        f="6*x[0]+6*x[1]",
        g="pow(x[0],3)+pow(x[1],3)",
    )
    A = np.array([[1, 0], [0, 1]])
    generator = DarcyGenerator(test_params)
    (u, p) = generator.solve_variational_form(A)
    # print(f"{np.max(u) = }")
    u_exact = fe.Expression(("3*pow(x[0],2)", "3*pow(x[1],2)"), degree=1)
    p_exact = fe.Expression("pow(x[0],3)+pow(x[1],3)", degree=0)
    # u_diff.vector()[:] = u.vector() - u_exact.vector()
    print("Finished")
