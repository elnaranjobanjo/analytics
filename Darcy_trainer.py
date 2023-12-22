import dolfin
import fenics as fe
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from dataclasses import dataclass, field

import Darcy_generator as Dg
import engine as ng

# Given a triplet [eig_1,eig_2,theta] the factory class
# trains a pair of darcy nets u,p solving
#       div u = f
#    A grad p = u
#           p = 0  b.c.
# with A sym. pos. def. with eigen vals eig_1 and eig_2
# The matrix of eigvecs is the rotation matrix generated by theta


class Darcy_nn(nn.Module):
    def __init__(
        self, input_size: int = 3, hidden_size: int = 64, output_size: int = 1
    ):
        super(Darcy_nn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x: torch.tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x


@dataclass
class DarcyPDELossParams:
    h: float = 0.1
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    f: str = "10"


class Darcy_PDE_Loss(nn.Module):
    def __init__(self, params: DarcyPDELossParams):
        super(Darcy_PDE_Loss, self).__init__()
        self.lossfun = nn.MSELoss()
        self.model_space = Dg.make_Darcy_model_space(params.mesh, params.degree)

        # Build the right-hand side vector L
        L = Dg.define_rhs(self.model_space, params.degree, params.f)
        L_np = fe.assemble(L).get_local()
        self.f = torch.from_numpy(L_np)

    def forward(
        self, u_dofs: torch.tensor, p_dofs: torch.tensor, A_matrix_params: list
    ):
        return self.lossfun(
            torch.matmul(
                self.assemble_system(A_matrix_params),
                torch.cat((u_dofs, p_dofs), dim=0),
            ),
            self.f,
        )

    def assemble_system(self, A_matrix_params: list):
        return torch.from_numpy(
            fe.assemble(
                Dg.define_linear_system(
                    Dg.get_A_matrix_from(A_matrix_params), self.model_space
                )
            ).array()
        )


class Darcy_nn_Solver:
    def __init__(self):
        pass

    def init_from_nets(
        self,
        u_net: Darcy_nn,
        p_net: Darcy_nn,
        model_space: fe.FunctionSpace,
    ):
        self.u_net = u_net
        self.p_net = p_net
        self.model_space = model_space
        return self

    def to_fenics(self, A_matrix_params: list):  # -> tuple(fe.Function, fe.Function):
        bias_term = torch.tensor([1], dtype=A_matrix_params.dtype)
        x_with_bias = torch.cat(
            (bias_term, torch.tensor(np.array(A_matrix_params))), dim=0
        )
        u_dofs = self.u.forward(x_with_bias).detach().numpy()
        p_dofs = self.p.forward(x_with_bias).detach().numpy()

        (u, p) = fe.Function(self.model_space).split()

        u.vector().set_local(u_dofs)
        p.vector().set_local(p_dofs)
        return (u, p)

    def save(self, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        mesh_file = dolfin.File(os.path.join(directory_path, "mesh.xml"))
        mesh_file << self.model_space.mesh()

        dict = {"degree": self.model_space.sub(0).ufl_element().degree()}
        with open(os.path.join(directory_path, "degree.json"), "w") as json_file:
            json.dump(dict, json_file)

        torch.save(self.u_net.state_dict(), os.path.join(directory_path, "u_net.pt"))
        torch.save(self.p_net.state_dict(), os.path.join(directory_path, "p_net.pt"))

    def load(self, directory_path: str):
        with open(os.path.join(directory_path, "degree.json"), "r") as json_file:
            degree_json = json.load(json_file)

        self.model_space = ng.make_Darcy_model_space(
            fe.Mesh(os.path.join(directory_path, "mesh.xml")), degree_json["degree"]
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.u_net = Darcy_nn(
            input_size=4,
            hidden_size=int((2 / 3) * (4 + self.model_space.sub(0).dim())),
            output_size=self.model_space.sub(0).dim(),
        )
        self.u_net.load_state_dict(torch.load(directory_path + "/u_net.pt"))
        self.u_net.to(device)
        self.p_net = Darcy_nn(
            input_size=4,
            hidden_size=int((2 / 3) * (4 + self.model_space.sub(1).dim())),
            output_size=self.model_space.sub(1).dim(),
        )
        self.p_net.load_state_dict(torch.load(directory_path + "/p_net.pt"))
        self.p_net.to(device)
        return self


@dataclass
class DarcynnFactoryParams:
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    f: str = "10"
    epochs: int = 100
    learn_rate: float = 0.001
    dataless: bool = True


class Darcy_nn_Factory:
    def __init__(self, params: DarcynnFactoryParams):
        PDE_loss_params = DarcyPDELossParams(
            mesh=params.mesh,
            degree=params.degree,
            f=params.f,
        )

        self.PDE_loss = Darcy_PDE_Loss(PDE_loss_params)

        self.epochs = params.epochs
        self.learn_rate = params.learn_rate
        self.input_size = 4
        self.u_output_size = self.PDE_loss.model_space.sub(0).dim()
        self.u_hidden_size = int((2 / 3) * (self.input_size + self.u_output_size))
        self.p_output_size = self.PDE_loss.model_space.sub(1).dim()
        self.p_hidden_size = int((2 / 3) * (self.input_size + self.p_output_size))
        self.dataless = params.dataless
        if not self.dataless:
            self.Data_loss = nn.MSELoss()

    def fit(
        self,
        training_data: list,
        validation_data: list,
        batch_size: int,
        output_dir: str,
        verbose: bool = False,
    ) -> Darcy_nn_Solver:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        u_net = Darcy_nn(
            input_size=self.input_size,
            hidden_size=self.u_hidden_size,
            output_size=self.u_output_size,
        ).to(device)

        p_net = Darcy_nn(
            input_size=self.input_size,
            hidden_size=self.p_hidden_size,
            output_size=self.p_output_size,
        ).to(device)

        u_optimizer = torch.optim.Adam(u_net.parameters(), lr=self.learn_rate)
        p_optimizer = torch.optim.Adam(p_net.parameters(), lr=self.learn_rate)

        if self.dataless:
            training_set = TensorDataset(torch.tensor(training_data[0]))
            validation_set = TensorDataset(torch.tensor(validation_data[0]))
        else:
            training_set = TensorDataset(
                torch.tensor(training_data[0]), torch.tensor(training_data[1])
            )
            validation_set = TensorDataset(
                torch.tensor(validation_data[0]), torch.tensor(validation_data[1])
            )
        training_loader = DataLoader(
            dataset=training_set, batch_size=batch_size, shuffle=False
        )
        validation_loader = DataLoader(
            dataset=validation_set, batch_size=batch_size, shuffle=False
        )

        losses = []
        for i in range(self.epochs):
            training_loss, validation_loss = self.one_grad_descent_iter(
                training_loader,
                validation_loader,
                u_net,
                p_net,
                u_optimizer,
                p_optimizer,
            )
            losses.append([training_loss, validation_loss])
            # with open(os.path.join(output_dir, "log.txt"), "a") as file:
            #     file.write(f"epoch number {i}")
            #     file.write(f"training loss = {training_loss}\n")
            #     file.write(f"validation loss = {validation_loss}\n")

            if verbose:
                print(f"epoch = {i+1}")
                print(f"training loss = {training_loss}")
                print(f"validation loss = {validation_loss}\n")
        if self.dataless:
            out_csv = pd.DataFrame(
                [[l[0][0], l[1][0]] for l in losses], columns=["training", "validation"]
            )
            out_csv.to_csv(os.path.join(output_dir, "loss.csv"), index=False)
        else:
            out_csv = pd.DataFrame(
                [
                    [l[0][0], l[0][1], l[0][2], l[1][0], l[1][1], l[1][2]]
                    for l in losses
                ],
                columns=[
                    "total_training",
                    "PDE_training_loss",
                    "Data_training_loss",
                    "total_validation",
                    "PDE_validation_loss",
                    "Data_validation_loss",
                ],
            )
            out_csv.to_csv(os.path.join(output_dir, "loss.csv"), index=False)

        return Darcy_nn_Solver().init_from_nets(u_net, p_net, self.PDE_loss.model_space)

    def one_grad_descent_iter(
        self, training_loader, validation_loader, u_net, p_net, u_optimizer, p_optimizer
    ):
        training_loss = [[], [], []]
        validation_loss = [[], [], []]

        u_net.train()
        p_net.train()
        for batch in training_loader:
            if self.dataless:
                x_batch = batch[0]
                total_loss = 0
                training_loss[2].append(0)
            else:
                x_batch, y_batch = batch
                Data_loss = self.calculate_Data_loss(u_net, p_net, x_batch, y_batch)
                total_loss = Data_loss
                training_loss[2].append(Data_loss.item())

            PDE_loss = self.calculate_PDE_loss(u_net, p_net, x_batch)
            total_loss = total_loss + PDE_loss
            training_loss[1].append(PDE_loss.item())
            training_loss[0].append(total_loss.item())

            total_loss.backward()
            u_optimizer.step()
            p_optimizer.step()

        for batch in validation_loader:
            if self.dataless:
                x_batch = batch[0]
                total_val_loss = 0
                validation_loss[2].append(0)
            else:
                x_batch, y_batch = batch
                Data_loss = self.calculate_Data_loss(u_net, p_net, x_batch, y_batch)
                total_val_loss = Data_loss
                validation_loss[2].append(Data_loss.item())

            PDE_loss = self.calculate_PDE_loss(u_net, p_net, x_batch)
            total_val_loss = total_val_loss + PDE_loss
            validation_loss[1].append(PDE_loss.item())
            validation_loss[0].append(total_val_loss.item())

        return [np.array(loss).mean(axis=0) for loss in training_loss], [
            np.array(loss).mean(axis=0) for loss in validation_loss
        ]

    def calculate_PDE_loss(self, u_net, p_net, x_batch):
        loss = 0
        for x in x_batch:
            bias_term = torch.tensor([1], dtype=x.dtype)
            x_with_bias = torch.cat((bias_term, x), dim=0)
            loss += self.PDE_loss(u_net(x_with_bias), p_net(x_with_bias), x)

        return loss / len(x_batch)

    def calculate_Data_loss(self, u_net, p_net, x_batch, y_batch):
        loss = 0
        for x, y in zip(x_batch, y_batch):
            bias_term = torch.tensor([1], dtype=x.dtype)
            x_with_bias = torch.cat((bias_term, x), dim=0)
            loss += self.Data_loss(
                torch.cat((u_net(x_with_bias), p_net(x_with_bias)), dim=0), y
            )

        return loss / len(x_batch)


@dataclass
class DarcyTrainingParams:
    mesh: fe.Mesh = fe.UnitSquareMesh(10, 10)
    degree: int = 1
    f: str = "10"
    A_matrix_params: list[list[float], list[float]] = field(
        default_factory=lambda: [[5, 10], [5, 10]]
    )
    epochs: int = 10
    learn_rate: float = 0.001
    dataless: bool = True
    number_of_data_points: int = 100
    percentage_for_validation: float = 0.2
    batch_size: int = 10
