from typing import Tuple
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


# Solves:
#       div u = f
#    A grad p = u
#           p = 0  b.c.
# Generates data for the sampling of the mapping A -> (u,p)


@dataclass
class DarcySimParams:
    h: float = 0.1
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    f: str = "1"


class DarcyGenerator:
    mesh: fe.Mesh
    degree: int
    f: fe.Expression
    model_space: fe.FunctionSpace

    def __init__(self, params: DarcySimParams):
        self.mesh = params.mesh
        self.degree = params.degree
        self.f = fe.Expression(params.f, degree=2)
        self.model_space = fe.FunctionSpace(
            self.mesh,
            fe.FiniteElement("BDM", self.mesh.ufl_cell(), self.degree)
            * fe.FiniteElement("DG", self.mesh.ufl_cell(), self.degree - 1),
        )

    def compute_solution(
        self, A: np.array, supress_fe_log: bool = True
    ) -> tuple[np.array, np.array]:
        (u, p) = self.solve_variational_form(A, supress_fe_log=supress_fe_log).split()
        return (u.vector().get_local(), p.vector().get_local())

    def solve_variational_form(
        self,
        A: np.array,
        supress_fe_log: bool = True,
    ) -> tuple[fe.Function, fe.Function]:
        (u, p) = fe.TrialFunctions(self.model_space)
        (v, q) = fe.TestFunctions(self.model_space)
        # Ainv = np.linalg.inv(A)
        # fe.Constant(((Ainv[0, 0], Ainv[0, 1]), (Ainv[1, 0], Ainv[1, 1]))) *
        a = (
            fe.dot(
                u,
                v,
            )
            + fe.div(v) * p
            + fe.div(u) * q
        ) * fe.dx
        L = self.f * q * fe.dx

        if supress_fe_log:
            fe.set_log_level(50)

        sol = fe.Function(self.model_space)
        fe.solve(
            a == L,
            sol
        )
        print(f"in gen {self.model_space.mesh().num_cells() = }")
        print(f"in gen {self.mesh.num_cells() = }")
        return sol