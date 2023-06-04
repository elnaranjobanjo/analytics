from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

h = 0.1
A = Constant(((1, 2), (3, 4)))
mesh = UnitSquareMesh(round(1/(h*np.sqrt(2))), round(1/(h*np.sqrt(2))))

# Define finite elements spaces and build mixed space
BDM = FiniteElement("BDM", mesh.ufl_cell(), 1, variant="integral")
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, BDM * DG)

# Define trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Define source function
f = Expression("1", degree=2)

# Define variational form

a = (dot(A * u, v) + div(v)*p + div(u)*q)*dx
L = - f*q*dx

# Define function G such that G \cdot n = g
class BoundarySource(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        #g = sin(5*x[0])
        g = 0
        values[0] = g*n[0]
        values[1] = g*n[1]
        
    def value_shape(self):
        return (2,)

G = BoundarySource(mesh, degree=2)

# Define essential boundary
def boundary(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

bc = DirichletBC(W.sub(0), G, boundary)

# Compute solution
w = Function(W)
solve(a == L, w, bc)
(u, p) = w.split()

# Plot sigma and u
plt.figure()
plot(u)

plt.figure()
plot(p)

plt.show()