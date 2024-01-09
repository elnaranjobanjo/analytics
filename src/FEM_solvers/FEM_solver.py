import fenics as fe
import numpy as np
from dataclasses import dataclass

import formulation as F


class Darcy_FEM_Solver:
    def __init__(self, params: F.formulation_params):
        if params.PDE == "Darcy_dual":
            self.formulation = F.D.Darcy_dual_formulation(params)
        elif params.PDE == "Darcy_primal":
            self.formulation = F.D.Darcy_primal_formulation(params)
        else:
            ValueError(f"The Darcy formulation {params.PDE} is not implemented")

    def compute_solution_eig_rep(
        self,
        A_matrix_params: list,
        supress_fe_log: bool = True,
    ):
        return self.compute_solution_from_A(
            F.D.get_A_matrix_from(A_matrix_params),
            supress_fe_log=supress_fe_log,
        )

    def compute_solution_from_A(
        self,
        A: np.array,
        supress_fe_log: bool = True,
    ) -> tuple[np.array, np.array]:
        return (
            self.solve_variational_form(A, supress_fe_log=supress_fe_log)
            .vector()
            .get_local()
        )

    def solve_variational_form(
        self,
        A: np.array,
        supress_fe_log: bool = True,
    ) -> fe.Function:
        if supress_fe_log:
            fe.set_log_level(50)

        return self.formulation.solve(A)
