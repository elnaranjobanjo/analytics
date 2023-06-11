from typing import Tuple
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
import random


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


class Darcy_Solver:
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

    def compute_solution_eig_rep(
        self,
        eigen_1: float,
        eigen_2: float,
        theta: float,
        supress_fe_log: bool = True,
        split: bool = False,
    ):
        return self.compute_solution_from_A(
            self.get_A_matrix_from(eigen_1, eigen_2, theta),
            supress_fe_log=supress_fe_log,
            split=split,
        )

    def get_A_matrix_from(self, eigen_1, eigen_2, theta):
        return (
            np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            * np.array([[eigen_1, 0], [0, eigen_2]])
            * np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
        )

    def compute_solution_from_A(
        self, A: np.array, supress_fe_log: bool = True, split: bool = False
    ) -> tuple[np.array, np.array]:
        if split:
            (u, p) = self.solve_variational_form(
                A, supress_fe_log=supress_fe_log
            ).split()
            return (u.vector().get_local(), p.vector().get_local())
        return (
            self.solve_variational_form(A, supress_fe_log=supress_fe_log)
            .vector()
            .get_local()
        )

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
        L = self.f * q * fe.dx

        if supress_fe_log:
            fe.set_log_level(50)

        sol = fe.Function(self.model_space)
        fe.solve(a == L, sol)
        return sol


def generate_data_using_eig_rep(
    params: DarcySimParams, eig_range: list, N: int, output_address: str
) -> None:
    matrix_params = [
        [
            random.uniform(eig_range[0], eig_range[1]),
            random.uniform(eig_range[0], eig_range[1]),
            random.uniform(0, 2 * np.pi),
        ]
        for _ in range(N)
    ]

    solver = Darcy_Solver(params)

    velocities_pressures = list(
        map(
            lambda x: solver.compute_solution_eig_rep(x[0], x[1], x[2]),
            matrix_params,
        )
    )

    pd.DataFrame(matrix_params).to_csv(output_address + "_X.csv")
    pd.DataFrame(velocities_pressures).to_csv(output_address + "_Y.csv")
