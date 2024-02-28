from abc import ABC
from dataclasses import dataclass, field
import fenics as fe
import numpy as np
import os


@dataclass
class formulation_params:
    PDE: str = "Darcy_primal"
    mesh_descr: str = "unitSquare10"
    degree: int = 1
    f: str = "10"


def print_formulation_params(params: formulation_params) -> None:
    print(f"PDE formulation specs:")
    print(f"PDE =  {params.PDE}")
    print(f"degree = {params.degree}")
    print(f"f = {params.f}\n")


def make_formulation_params_dataclass(params_dict: dict) -> formulation_params:
    params = formulation_params()
    for key, value in params_dict.items():
        if key == "PDE":
            params.PDE = value
        elif key == "mesh_descr":
            params.mesh_descr = value
        elif key == "degree":
            params.degree = value
        elif key == "f":
            params.f = value
        else:
            raise ValueError(f"The key {key} is not a formulation parameter.")
    return params


def make_mesh(mesh_descr: str) -> fe.Mesh:
    if mesh_descr == "unitSquare10":
        return fe.UnitSquareMesh(10, 10)
    elif mesh_descr == "unitSquare5":
        return fe.UnitSquareMesh(5, 5)
    elif os.path.exists(mesh_descr):
        return fe.Mesh(mesh_descr)

    else:
        raise ValueError(f"The mesh {mesh_descr} is not implemented")


class PDE_formulation(ABC):
    def __init__(self):
        pass

    def initialize_function(self) -> fe.Function:
        return fe.Function(self.model_space)

    def get_mesh(self) -> fe.Mesh:
        return self.model_space.mesh()

    def get_model_space(self) -> fe.FunctionSpace:
        return self.model_space

    def compute_multiple_actions_on(
        self, X: np.array, A_matrix_params: np.array
    ) -> np.array:
        return np.transpose(
            np.array(
                list(
                    map(
                        lambda x: self.compute_single_action_on(*x),
                        zip(X, A_matrix_params),
                    )
                )
            )
        )


# import sys

# sys.path.append("./src/formulations/PDEs/")

import src.formulations.PDEs.Darcy as D
