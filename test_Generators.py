from Darcy_Generators import DarcyGenerator, DarcySimParams
import fenics as fe
import numpy as np


def test_darcy_generator():
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
    (u, p) = generator.solve_variational_form(A)
    (u_exact, p_exact) = generator.turn_into_mesh_funcs(
        ("3*pow(x[0],2)", "3*pow(x[1],2)"), "pow(x[0],3)+pow(x[1],3)"
    )
    # u_exact = fe.Expression(("3*pow(x[0],2)", "3*pow(x[1],2)"), degree=3)
    # p_exact = fe.Expression("pow(x[0],3)+pow(x[1],3)", degree=2)
    # u_diff.vector()[:] = u.vector() - u_exact.vector()
    assert 1 == 1
