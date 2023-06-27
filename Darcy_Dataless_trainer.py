import numpy as np
import fenics as fe
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from dataclasses import dataclass, field

# Given a triplet [eig_1,eig_2,theta] the factory class
# trains a pair of darcy nets u,p solving
#       div u = f
#    A grad p = u
#           p = 0  b.c.
# with A sym. pos. def. with eigen vals eig_1 and eig_2
# The matrix of eigvecs is the rotation matrix generated by theta


class Darcy_nn(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1):
        super(Darcy_nn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x


class Darcy_Energy_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.tensor(1)


class Test_Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        # Save inputs for backward
        ctx.save_for_backward(x, y)
        # Compute loss using numpy (or any other library)
        loss_np = (x.detach().numpy() - np.pi) ** 2 + (
            y.detach().numpy() - 2 * np.pi
        ) ** 2
        return torch.tensor(loss_np, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        # Compute gradients
        x, y = ctx.saved_tensors
        grad_x = 2 * (x.detach().numpy() - np.pi)
        grad_y = 2 * (y.detach().numpy() - 2 * np.pi)
        return torch.tensor(grad_x), torch.tensor(grad_y)


@dataclass
class DatalessDarcyTrainingParams:
    h: float = 0.1
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    f: str = "10"
    A_matrix_params: list[float] = field(default_factory=list)  # eig_1,eig_2,theta
    epochs: int = 10
    learn_rate: float = 0.001


def get_matrix_params_from(A: np.array) -> list:
    eig_vals, eig_vecs = np.linalg.eig(A)
    # When A is symmetric eig_vecs is a rotation matrix
    cos_theta = eig_vecs[0, 0]
    sin_theta = eig_vecs[1, 0]
    return [eig_vals[0], eig_vals[1], np.arctan2(sin_theta, cos_theta)]


class DatalessDarcy_solver:
    def __init__(self, params):
        pass


class DatalessDarcy_nn_Factory:
    def __init__(self, params: DatalessDarcyTrainingParams):
        self.init_FEM_formulation(params)
        self.init_training_settings(params)

    def init_FEM_formulation(self, params: DatalessDarcyTrainingParams):
        self.A_matrix_params = params.A_matrix_params

        self.mesh = params.mesh
        self.f = fe.Expression(params.f, degree=params.degree - 1)
        # Hdiv-L2 conforming FE space.
        self.model_space = fe.FunctionSpace(
            self.mesh,
            fe.FiniteElement("BDM", self.mesh.ufl_cell(), params.degree)
            * fe.FiniteElement("DG", self.mesh.ufl_cell(), params.degree - 1),
        )

    def init_training_settings(self, params: DatalessDarcyTrainingParams):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.epochs = params.epochs
        self.learn_rate = params.learn_rate
        self.loss = Test_Loss.apply
        # input_size = 2 * self.mesh.num_vertices() +1
        # output_size = 10
        # hidden_size = 64

    def fit(self, verbose: bool = False) -> DatalessDarcy_solver:
        # The only input is the bias
        u_net = Darcy_nn(input_size=1)
        p_net = Darcy_nn(input_size=1)
        u_optimizer = torch.optim.Adam(u_net.parameters(), lr=self.learn_rate)
        p_optimizer = torch.optim.Adam(p_net.parameters(), lr=self.learn_rate)
        physics_loss = Darcy_Energy_Loss()

        u_net.train()
        p_net.train()

        loss = self.one_grad_descent_iter(u_net, p_net, u_optimizer, p_optimizer)
        for i in range(self.epochs):
            loss = self.one_grad_descent_iter(u_net, p_net, u_optimizer, p_optimizer)
            if verbose:
                print(f"Current_Action = {loss}")

        bias = torch.tensor([1.0])
        print(f"{u_net(bias).item() = }")
        print(f"{p_net(bias).item() = }")

        return DatalessDarcy_solver([])

    def one_grad_descent_iter(self, u, p, u_optimizer, p_optimizer):
        bias = torch.tensor([1.0])
        # Coordinates in the FEM basis self.model_space
        # u_coordinates = np.array(u(bias).item())
        # p_coordinates = np.array(p(bias).item())
        x = u(bias)  # .detach().numpy()
        y = p(bias)  # .detach().numpy()

        # system_state = fe.Function(self.model_space)
        # (u, p) = system_state.split()

        # gradients_array = np.matrix([2 * (x - np.pi), 2 * (y - 2 * np.pi)])
        # loss = torch.tensor(
        #     torch.tensor((x - np.pi) ** 2 + (y - 2 * np.pi) ** 2), requires_grad=True
        # )
        # gradients = torch.tensor(gradients_array, requires_grad=False)

        # # Perform the backward pass with the manually computed gradients
        # loss.backward(gradients)

        loss = self.loss(x, y)
        loss.backward()

        # Use an optimizer to update your parameters
        u_optimizer.step()
        p_optimizer.step()

        return loss.item()


if __name__ == "__main__":
    h = 0.25
    u_expression = (
        "5*x[1]*(1-x[1])*(1-2*x[0])+x[0]*(1-x[0])*(1-2*x[1])",
        "5*x[0]*(1-x[0])*(1-2*x[1])+x[1]*(1-x[1])*(1-2*x[0])",
    )
    p_expression = "x[0]*x[1]*(1-x[0])*(1-x[1])"

    exact_solution = [u_expression, p_expression]
    test_params = DatalessDarcyTrainingParams(
        h=h,
        mesh=fe.UnitSquareMesh(
            round(1.0 / (h * np.sqrt(2.0))),
            round(1.0 / (h * np.sqrt(2.0))),
        ),
        degree=5,
        f="-10*x[1]*(1-x[1])-10*x[0]*(1-x[0])+2*(1-2*x[0])*(1-2*x[1])",
        A_matrix_params=get_matrix_params_from(np.array([[5.0, 1.0], [1.0, 5.0]])),
        epochs=10000,
        learn_rate=0.00001,
    )

    sol_factory = DatalessDarcy_nn_Factory(test_params)
    solver = sol_factory.fit(verbose=True)

    # sol = sol_factory.get_nets()
