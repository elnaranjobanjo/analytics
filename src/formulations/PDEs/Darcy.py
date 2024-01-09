import fenics as fe
import numpy as np

import formulation as F


def get_matrix_params_from(A: np.array) -> list:
    eig_vals, eig_vecs = np.linalg.eig(A)
    # When A is symmetric eig_vecs is a rotation matrix
    # However, different selection for the eigen vectors will result in different rotation angles
    cos_theta = eig_vecs[0, 0]
    sin_theta = -eig_vecs[1, 0]
    return [eig_vals[0], eig_vals[1], np.arctan2(sin_theta, cos_theta)]


def get_A_matrix_from(A_matrix_params: list, verbose=False):
    R = np.array(
        [
            [np.cos(A_matrix_params[2]), -np.sin(A_matrix_params[2])],
            [np.sin(A_matrix_params[2]), np.cos(A_matrix_params[2])],
        ]
    )
    return np.matmul(
        np.matmul(R, np.array([[A_matrix_params[0], 0], [0, A_matrix_params[1]]])), R.T
    )


# Encodes the variational formulation for:
#       -div u = f
#    A grad p = u
#           p = 0  b.c.
# Intended for the generation of data for the sampling of the mapping A -> (u,p)
class Darcy_dual_formulation(F.PDE_formulation):
    def __init__(self, params: F.formulation_params):
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
class Darcy_primal_formulation(F.PDE_formulation):
    def __init__(self, params: F.formulation_params):
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
