from abc import ABC, abstractmethod
from dataclasses import dataclass
import fenics as fe


@dataclass
class formulation_params:
    PDE: str = "Darcy_primal"
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
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
        elif key == "mesh":
            if value[0] == "unit_square":
                params.mesh = fe.UnitSquareMesh(value[1], value[1])
            else:
                raise ValueError(f"The mesh type {value[0]} is not implemented")
        elif key == "degree":
            params.degree = value
        elif key == "f":
            params.f = value
        else:
            raise ValueError(f"The key {key} is not a formulation")
    return params


class PDE_formulation(ABC):
    def __init__(self):
        pass

    def initialize_function(self) -> fe.Function:
        return fe.Function(self.model_space)

    def get_mesh(self) -> fe.Mesh:
        return self.model_space.mesh()

    def get_model_space(self) -> fe.FunctionSpace:
        return self.model_space


import sys

sys.path.append("./src/formulations/PDEs/")

import Darcy as D
