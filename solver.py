from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

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

# Define essential boundary
def boundary(x):
    #return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
    return False 

def compute_velocities(h,material_prop, f):
    A = Constant(((material_prop[0,0], material_prop[0,1]), (material_prop[1,0], material_prop[1,1])))
    mesh = UnitSquareMesh(round(1/(h*np.sqrt(2))), round(1/(h*np.sqrt(2))))

    # Define finite elements spaces and build mixed space
    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1, variant="integral")
    DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, BDM * DG)

    # Define trial and test function
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # Define variational form

    a = (dot(A * u, v) + div(v)*p + div(u)*q)*dx
    L = - f*q*dx

    G = BoundarySource(mesh, degree=2)

    bc = DirichletBC(W.sub(0), G, boundary)

    # Compute solution
    w = Function(W)
    solve(a == L, w, bc)
    (u,p) = w.split()
    return (u,p)

def plot_up(u,p):
    # Plot sigma and u
    plt.figure()
    plot(u)

    plt.figure()
    plot(p)

    plt.show()

def make_into_np_ndarray(u,p):
    return (u.vector().get_local(),p.vector().get_local())

if __name__ == '__main__':
   material_matrix = np.array([[1,0],[0,1]])
   # Define source function
   f = Expression("1", degree=2)
   h = 0.1
   (u,p) = compute_velocities(h,material_matrix,f)
   #(nd_u,nd_p) = make_into_np_ndarray(u,p)
   #print(f'{np.max(nd_u ) = }')
   plot_up(u,p)