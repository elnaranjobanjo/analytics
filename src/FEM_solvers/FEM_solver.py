from dataclasses import dataclass, field
import fenics as fe
import numpy as np
import os
import pandas as pd
import random

import src.formulations.formulation as F


class Darcy_FEM_Solver:
    def __init__(self, params: F.formulation_params):
        if params.PDE == "Darcy_dual":
            self.formulation = F.D.Darcy_dual_formulation(params)
        elif params.PDE == "Darcy_primal":
            self.formulation = F.D.Darcy_primal_formulation(params)
        else:
            raise ValueError(f"The Darcy formulation {params.PDE} is not implemented")

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


@dataclass
class data_gen_params:
    num_data_points: dict = ({"training": 100, "validation": 50},)
    A_matrix_params: list = field(default_factory=lambda: [[5, 10], [5, 10]])


def print_data_gen_params(params: data_gen_params):
    print(f"Data generation specs:")
    print(f"number_of_data_points = {params.num_data_points}")
    print(f"A_matrix_params = {params.A_matrix_params}\n")


def make_data_gen_params_dataclass(params_dict: dict) -> data_gen_params:
    params = data_gen_params()
    for key, value in params_dict.items():
        if key == "num_data_points":
            params.num_data_points = value
        elif key == "A_matrix_params":
            params.A_matrix_params = value
        else:
            raise ValueError(f"The key {key} is not a data generation parameter")
    return params


def generate_data(
    gen_params: data_gen_params,
    formulation_params: F.formulation_params,
    output_dir: str = None,
    include_output_vals: bool = True,
    save_in_csv: bool = True,
):
    output_data = []
    for data_use_type, number_of_data_points in gen_params.num_data_points.items():
        X = [
            [
                random.uniform(
                    gen_params.A_matrix_params[0][0], gen_params.A_matrix_params[0][1]
                ),
                random.uniform(
                    gen_params.A_matrix_params[1][0], gen_params.A_matrix_params[1][1]
                ),
                random.uniform(0, 2 * np.pi),
            ]
            for _ in range(number_of_data_points)
        ]

        csv = pd.DataFrame(X, columns=["eig_1", "eig_2", "theta"])

        if include_output_vals:
            FEM_solver = Darcy_FEM_Solver(formulation_params)
            Y = np.array(
                list(
                    map(
                        lambda x: FEM_solver.compute_solution_eig_rep(x),
                        X,
                    )
                )
            )

            data = [X, Y]
            column_labels = make_column_labels(
                formulation_params.PDE,
                FEM_solver.formulation.get_model_space(),
            )

            Y_csv = pd.DataFrame(Y, columns=column_labels)

            csv = pd.concat([csv, Y_csv], axis=1)
        else:
            data = [X]

        output_data.append(data)
        if save_in_csv:
            csv.to_csv(os.path.join(output_dir, data_use_type + ".csv"), index=False)

    return output_data


def make_column_labels(PDE: str, model_space: fe.FunctionSpace) -> list:
    if PDE == "Darcy_primal":
        num_p_dofs = model_space.dim()
        column_labels = [f"p_{i}" for i in range(num_p_dofs)]
    elif PDE == "Darcy_dual":
        num_u_dofs = model_space.sub(0).dim()
        num_p_dofs = model_space.sub(1).dim()
        column_labels = [f"u_{i}" for i in range(num_u_dofs)] + [
            f"p_{i}" for i in range(num_p_dofs)
        ]
    else:
        raise ValueError(f"The PDE {PDE} is not implemented")
    return column_labels
