from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import dolfin
import fenics as fe
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import sys

import formulation as F


@dataclass
class training_params:
    formulation_params: F.formulation_params = F.formulation_params()
    A_matrix_params: list[list[float], list[float]] = field(
        default_factory=lambda: [[5, 10], [5, 10]]
    )
    epochs: int = 10
    learn_rate: float = 0.001
    losses_to_use: list[str] = field(default_factory=lambda: [["PDE"], ["data"]])
    number_of_data_points: int = 100
    percentage_for_validation: float = 0.2
    batch_size: int = 10


def dense_net_size_heuristic(input_size: int, output_size: int) -> int:
    return int((2 / 3) * (input_size + output_size))


class dense_nn(nn.Module):
    def __init__(
        self, input_size: int = 4, hidden_size: int = 64, output_size: int = 1
    ):
        super(dense_nn, self).__init__()
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


class nn_solver(ABC):
    def __init__(self):
        pass

    def init_from_nets(self, nets: dict, model_space: fe.FunctionSpace):
        self.nets = nets
        self.model_space = model_space
        return self

    def save(
        self,
        directory_path: str,
    ) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        mesh_file = dolfin.File(os.path.join(directory_path, "mesh.xml"))
        mesh_file << self.model_space.mesh()
        dict = {"degree": self.model_space.ufl_element().degree()}
        with open(os.path.join(directory_path, "degree.json"), "w") as json_file:
            json.dump(dict, json_file)

        for net_name, net in self.nets.items():
            torch.save(net.state_dict(), os.path.join(directory_path, net_name + ".pt"))

    def load_mesh_and_device(self, directory_path: str) -> (fe.Mesh, int, torch.device):
        with open(os.path.join(directory_path, "degree.json"), "r") as json_file:
            degree_json = json.load(json_file)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return (
            fe.Mesh(os.path.join(directory_path, "mesh.xml")),
            degree_json["degree"],
            device,
        )


class PDE_trainer(ABC):
    def __init__(self, params):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.data_loss = nn.MSELoss()
        self.PDE_loss_type = nn.MSELoss()

        if len(params.losses_to_use) > 2:
            ValueError(
                f"There are more than two losses, the only losses implemented are PDE and data losses"
            )
        if "PDE" and "data" in params.losses_to_use:

            def losses(batch):
                x_batch, y_batch = batch
                return [
                    self.calculate_PDE_loss(x_batch),
                    self.calculate_data_loss(x_batch, y_batch),
                ]

        elif "PDE" in params.losses_to_use and "data" not in params.losses_to_use:

            def losses(batch):
                return [self.calculate_PDE_loss(batch[0]), 0]

        elif "data" in params.losses_to_use and "PDE" not in params.losses_to_use:

            def losses(batch):
                x_batch, y_batch = batch
                return [
                    0,
                    self.calculate_data_loss(x_batch, y_batch),
                ]

        else:
            ValueError(
                f"The losses to use {params.losses_to_use} has one type not implemented"
            )
        self.losses = losses

        return device

    def multiple_net_eval(self, x: torch.tensor) -> torch.tensor:
        return torch.cat(
            [
                net(
                    torch.cat(
                        (
                            torch.ones(x.shape[0], 1),
                            x,
                        ),
                        dim=1,
                    )
                )
                for net in self.nets.values()
            ],
            dim=1,
        )

    def single_net_eval(self, x):
        return torch.cat(
            [
                net(
                    torch.cat(
                        (
                            torch.tensor([1]),
                            x,
                        )
                    )
                )
                for net in self.nets.values()
            ],
            dim=0,
        )

    @abstractmethod
    def calculate_PDE_loss(self):
        pass

    def calculate_data_loss(
        self, x_batch: torch.tensor, y_batch: torch.tensor
    ) -> torch.tensor:
        return self.data_loss(
            self.multiple_net_eval(x_batch),
            y_batch,
        )

    @abstractmethod
    def get_nn_solver(self):
        pass

    def define_optimizers(self, learn_rate):
        self.optimizers = {}
        for net_name, net in self.nets.items():
            self.optimizers[net_name] = torch.optim.Adam(
                net.parameters(), lr=learn_rate
            )

    def train(self):
        for net in self.nets.values():
            net.train()

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()


sys.path.append("./src/trainers/PDEs/")
import Darcy_trainers as Dt


class nn_Factory:
    def __init__(self, params: training_params):
        if params.formulation_params.PDE == "Darcy_primal":
            self.trainer = Dt.Darcy_primal_trainer(params)
        elif params.formulation_params.PDE == "Darcy_dual":
            self.trainer = Dt.Darcy_dual_trainer(params)
        else:
            ValueError(f"The PDE {params.PDE} is not implemented")

        self.epochs = params.epochs
        self.dataless = "data" not in params.losses_to_use

    def fit(
        self,
        training_data: list,
        validation_data: list,
        batch_size: int,
        output_dir: str,
        verbose: bool = False,
    ):
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
            )

            losses.append([training_loss, validation_loss])

            if verbose:
                print(f"epoch = {i+1}")
                print(f"training loss = {training_loss}")
                print(f"validation loss = {validation_loss}\n")

        pd.DataFrame(
            [[l[0][0], l[0][1], l[0][2], l[1][0], l[1][1], l[1][2]] for l in losses],
            columns=[
                "total_training",
                "PDE_training_loss",
                "Data_training_loss",
                "total_validation",
                "PDE_validation_loss",
                "Data_validation_loss",
            ],
        ).to_csv(os.path.join(output_dir, "loss.csv"), index=False)

        return self.trainer.get_nn_solver()

    def one_grad_descent_iter(
        self,
        training_loader,
        validation_loader,
    ):
        training_loss = [[], [], []]
        validation_loss = [[], [], []]

        self.trainer.train()
        for batch in training_loader:
            self.trainer.zero_grad()
            losses = self.trainer.losses(batch)
            total_loss = sum(losses)
            total_loss.backward()
            self.trainer.step()
            training_loss[0].append(total_loss.item())
            training_loss[1].append(losses[0].item())
            training_loss[2].append(losses[1].item())

        for batch in validation_loader:
            losses = self.trainer.losses(batch)
            total_loss = sum(losses)

            validation_loss[0].append(total_loss.item())
            validation_loss[1].append(losses[0].item())
            validation_loss[2].append(losses[1].item())

        return [np.array(loss).mean(axis=0) for loss in training_loss], [
            np.array(loss).mean(axis=0) for loss in validation_loss
        ]
