from typing import Tuple
import fenics as fe
import numpy as np
import pandas as pd
from dataclasses import dataclass


# Solves:
#       div u = f
#    A grad p = u
#           p = 0  b.c.
# Generates data for the sampling of the mapping A -> (u,p)


@dataclass
class DarcySimParams:
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    f: str = "1"


def make_Darcy_model_space(mesh: fe.Mesh, degree: int) -> fe.FunctionSpace:
    # Hdiv-L2 conforming FE space.
    return fe.FunctionSpace(
        mesh,
        fe.FiniteElement("BDM", mesh.ufl_cell(), degree)
        * fe.FiniteElement("DG", mesh.ufl_cell(), degree - 1),
    )


def define_linear_system(A: np.array, model_space: fe.FunctionSpace) -> fe.Matrix:
    (u, p) = fe.TrialFunctions(model_space)
    (v, q) = fe.TestFunctions(model_space)
    return (
        fe.dot(
            fe.Constant(np.linalg.inv(A)) * u,
            v,
        )
        + fe.div(v) * p
        + fe.div(u) * q
    ) * fe.dx


def define_rhs(model_space: fe.FunctionSpace, degree: int, f: str):
    (v, q) = fe.TestFunctions(model_space)
    return fe.Expression(f, degree=degree - 1) * q * fe.dx


class Darcy_FEM_Solver:
    degree: int
    f: str
    model_space: fe.FunctionSpace

    def __init__(self, params: DarcySimParams):
        self.degree = params.degree
        self.model_space = make_Darcy_model_space(params.mesh, params.degree)
        self.L = define_rhs(self.model_space, params.degree, params.f)

    def compute_solution_eig_rep(
        self,
        A_matrix_params: list,
        supress_fe_log: bool = True,
        split: bool = False,
    ):
        return self.compute_solution_from_A(
            get_A_matrix_from(A_matrix_params),
            supress_fe_log=supress_fe_log,
            split=split,
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
        a = define_linear_system(A, self.model_space)
        if supress_fe_log:
            fe.set_log_level(50)

        sol = fe.Function(self.model_space)
        fe.solve(a == self.L, sol)
        return sol


def get_matrix_params_from(A: np.array) -> list:
    eig_vals, eig_vecs = np.linalg.eig(A)
    # When A is symmetric eig_vecs is a rotation matrix
    cos_theta = eig_vecs[0, 0]
    sin_theta = eig_vecs[1, 0]
    return [eig_vals[0], eig_vals[1], np.arctan2(sin_theta, cos_theta)]


def get_A_matrix_from(A_matrix_params: list):
    eigen_1 = A_matrix_params[0]
    eigen_2 = A_matrix_params[1]
    theta = A_matrix_params[2]
    return (
        np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        * np.array([[eigen_1, 0], [0, eigen_2]])
        * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    )


def generate_data_using_eig_rep(
    params: DarcySimParams, generated_A_matrix_params: list
) -> None:
    solver = Darcy_FEM_Solver(params)

    velocities_pressures = list(
        map(
            lambda x: solver.compute_solution_eig_rep(x),
            generated_A_matrix_params,
        )
    )

    return velocities_pressures
