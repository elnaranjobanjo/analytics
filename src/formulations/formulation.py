from abc import ABC, abstractmethod
from dataclasses import dataclass
import fenics as fe


@dataclass
class formulation_params:
    PDE: str = "Darcy_primal"
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    f: str = "10"


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
