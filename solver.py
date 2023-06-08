import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


# Solves:
#       div u = f
#    A grad p = u
#           u = 0  b.c
# Generates data for the sampling of the mapping A -> (u,p)
class DarcyGenerator:
    def __init__(self, params):
        self.mesh = params.mesh
        self.degree = params.degree
        self.g = fe.Expression(params.g, degree=params.degree)
        self.f = fe.Expression(params.f, degree=params.degree - 1)

    def find_velocities(
        self,
        A,
        if_plot=False,
        supress_fe_log=True,
        out_format="fenics",
    ):
        BDM = fe.FiniteElement("BDM", self.mesh.ufl_cell(), self.degree)
        DG = fe.FiniteElement("DG", self.mesh.ufl_cell(), self.degree - 1)
        W = fe.FunctionSpace(self.mesh, BDM * DG)

        (u, p) = fe.TrialFunctions(W)
        (v, q) = fe.TestFunctions(W)

        a = (
            fe.dot(
                fe.Constant(
                    (
                        (A[0, 0], A[0, 1], A[0, 2]),
                        (A[1, 0], A[1, 1], A[1, 2]),
                        (A[2, 0], A[2, 1], A[2, 2]),
                    )
                )
                * u,
                v,
            )
            + fe.div(v) * p
            + fe.div(u) * q
        ) * fe.dx
        L = -self.f * q * fe.dx

        w = fe.Function(W)

        if supress_fe_log:
            fe.set_log_level(50)

        fe.solve(a == L, w, fe.DirichletBC(W, self.g, "on_boundary"))
        (u, p) = w.split()

        # if if_plot:
        #    self.plot(u, p)

        if out_format == "fenics":
            return w.split()
        if out_format == "numpy":
            return (u.vector().get_local(), p.vector().get_local())


@dataclass
class DarcySimParams:
    h: float = 0.1
    mesh: fe.Mesh = fe.UnitCubeMesh(10, 10, 10)
    degree: int = 1
    g: str = ("0", "0", "0")
    f: str = "1"


if __name__ == "__main__":
    h = 0.1
    test_params = DarcySimParams(
        h=h,
        mesh=fe.UnitCubeMesh(
            round(1 / (h * np.sqrt(2))),
            round(1 / (h * np.sqrt(2))),
            round(1 / (h * np.sqrt(2))),
        ),
        f="1",
        g=("0", "0", "0"),
    )
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    generator = DarcyGenerator(test_params)
    (u, p) = generator.find_velocities(A, if_plot="false", out_format="fenics")
    print("Finished")
