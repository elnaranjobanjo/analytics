import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


# Solves:
#       div u = f
#    A grad p = u
#           p = g  b.c
# Generates data for the sampling of the mapping A -> (u,p)


@dataclass
class DarcySimParams:
    h: float = 0.1
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    g: str = "0"
    f: str = "1"


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

        w = fe.Function(W)

        if supress_fe_log:
            fe.set_log_level(50)

        fe.solve(a == L, w, fe.DirichletBC(W.sub(1), self.g, "on_boundary"))
        (u, p) = w.split()

        # if if_plot:
        #    self.plot(u, p)

        if out_format == "fenics":
            return w.split()
        if out_format == "numpy":
            return (u.vector().get_local(), p.vector().get_local())


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
    (u, p) = generator.find_velocities(A, if_plot="false", out_format="numpy")
    print(f"{np.max(u) = }")
    u_exact = fe.Expression(("3*pow(x[0],2)", "3*pow(x[1],2)"), degree=1)
    p_exact = fe.Expression("pow(x[0],3)+pow(x[1],3)", degree=0)
    # u_diff.vector()[:] = u.vector() - u_exact.vector()
    print("Finished")
