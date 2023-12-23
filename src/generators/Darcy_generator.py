import fenics as fe
import numpy as np
from dataclasses import dataclass


@dataclass
class DarcySimParams:
    formulation: str = "primal"
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    f: str = "10"


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


class PDE_formulation:
    def __init__(self):
        pass

    def initialize_function(self) -> fe.Function:
        return fe.Function(self.model_space)

    def get_mesh(self) -> fe.Mesh:
        return self.model_space.mesh()

    def get_model_space(self) -> fe.FunctionSpace:
        return self.model_space

    # def get_source_Lnp(self) -> np.array:
    #     return fe.assemble(self.L).get_local()


# Encodes the variational formulation for:
#       -div u = f
#    A grad p = u
#           p = 0  b.c.
# Intended for the generation of data for the sampling of the mapping A -> (u,p)
class Darcy_dual_formulation(PDE_formulation):
    def __init__(self, params: DarcySimParams):
        super().__init__()
        self.f = params.f
        self.degree = params.degree
        self.model_space = self.define_model_space(params.mesh)
        (self.u, self.p) = fe.TrialFunctions(self.model_space)
        (self.v, self.q) = fe.TestFunctions(self.model_space)
        self.L = self.define_rhs()

    def define_model_space(
        self,
        mesh: fe.Mesh,
    ) -> fe.FunctionSpace:
        # Hdiv-L2 conforming FE space.
        return fe.FunctionSpace(
            mesh,
            fe.FiniteElement("BDM", mesh.ufl_cell(), self.degree)
            * fe.FiniteElement("DG", mesh.ufl_cell(), self.degree - 1),
        )

    def define_linear_system(
        self,
        A: np.array,
    ) -> fe.Matrix:
        return (
            fe.dot(
                fe.Constant(np.linalg.inv(A)) * self.u,
                self.v,
            )
            + fe.div(self.v) * self.p
            - fe.div(self.u) * self.q
        ) * fe.dx

    def define_rhs(self) -> fe.Function:
        return fe.Expression(self.f, degree=self.degree - 1) * self.q * fe.dx

    def get_rhs_vector(self) -> np.array:
        return fe.assemble(self.L).array()

    def assemble_linear_system(self, A_matrix_params: list) -> np.array:
        return fe.assemble(
            self.define_linear_system(get_A_matrix_from(A_matrix_params))
        ).array()

    def solve(self, A: np.array) -> fe.Function:
        sol = self.initialize_function()
        fe.solve(self.define_linear_system(A) == self.L, sol)
        return sol


# Encodes the variational formulation for:
#       -div A grad p = f
#           p = 0  b.c.
# Intended for the generation of data for the sampling of the mapping A -> p
class Darcy_primal_formulation(PDE_formulation):
    def __init__(self, params):
        super().__init__()
        self.f = params.f
        self.degree = params.degree
        self.model_space = self.define_model_space(params.mesh)
        self.bc = self.define_bc()
        self.p = fe.TrialFunction(self.model_space)
        self.q = fe.TestFunction(self.model_space)
        self.L = self.define_rhs()

    def define_model_space(self, mesh: fe.Mesh):
        # H1 conforming space
        return fe.FunctionSpace(mesh, "Lagrange", self.degree)

    def define_linear_system(
        self,
        A: np.array,
    ) -> fe.Matrix:
        return fe.inner(fe.Constant(A) * fe.grad(self.p), fe.grad(self.q)) * fe.dx

    def define_rhs(self):
        return fe.Expression(self.f, degree=self.degree) * self.q * fe.dx

    def define_bc(self):
        # Define Dirichlet boundary (x,y = 0 or x,y = 1)
        def boundary(x):
            return (
                x[0] < fe.DOLFIN_EPS
                or x[0] > 1.0 - fe.DOLFIN_EPS
                or x[1] < fe.DOLFIN_EPS
                or x[1] > 1.0 - fe.DOLFIN_EPS
            )

        return fe.DirichletBC(self.model_space, fe.Constant(0.0), boundary)

    def assemble_linear_system(self, A_matrix_params: list) -> np.array:
        A = fe.assemble(self.define_linear_system(get_A_matrix_from(A_matrix_params)))
        self.bc.apply(A)
        return A.array()

    def get_rhs_vector(self) -> np.array:
        b = fe.assemble(self.L)
        self.bc.apply(b)
        return b.get_local()

    def solve(self, A: np.array) -> fe.Function:
        sol = self.initialize_function()
        fe.solve(self.define_linear_system(A) == self.L, sol, self.bc)
        return sol


class Darcy_FEM_Solver:
    def __init__(self, params: DarcySimParams):
        if params.formulation == "dual":
            self.formulation = Darcy_dual_formulation(params)
        elif params.formulation == "primal":
            self.formulation = Darcy_primal_formulation(params)
        else:
            return ValueError(
                f"The formulation {params.formulation} is not implemented"
            )

    def compute_solution_eig_rep(
        self,
        A_matrix_params: list,
        supress_fe_log: bool = True,
    ):
        return self.compute_solution_from_A(
            get_A_matrix_from(A_matrix_params),
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
