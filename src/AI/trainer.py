from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import dolfin
import fenics as fe
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import sys

import formulation as F
import neural_networks as nn


@dataclass
class training_params:
    formulation_params: F.formulation_params = F.formulation_params()
    nn_params: nn.nn_params = nn.nn_params()
    A_matrix_params: list[list[float], list[float]] = field(
        default_factory=lambda: [[5, 10], [5, 10]]
    )
    epochs: int = 10
    learn_rate: float = 0.001
    losses_to_use: list[str] = field(default_factory=lambda: [["PDE"], ["data"]])
    number_of_data_points: int = 100
    percentage_for_validation: float = 0.2
    batch_size: int = 10


def print_training_params(params: training_params) -> None:
    F.print_formulation_params(params.formulation_params)
    nn.print_neural_net_params(params.nn_params)
    print(f"Training specs:")
    print(f"A matrix params = {params.A_matrix_params}")
    print(f"epochs = {params.epochs}")
    print(f"learn rate = {params.learn_rate}")
    print(f"losses_to_use = {params.losses_to_use}")
    print(f"number of data points = {params.number_of_data_points}")
    print(f"percentage set for validation = {params.percentage_for_validation}")
    print(f"batch size = {params.batch_size}\n")


def make_training_params_dataclass(
    formulation_dict: dict, nn_dict: dict, params_dict: dict
) -> training_params:
    params = training_params(
        formulation_params=F.make_formulation_params_dataclass(formulation_dict),
        nn_params=nn.make_nn_params_dataclass(nn_dict),
    )
    for key, value in params_dict.items():
        if key == "A_matrix_params":
            params.A_matrix_params = value
        elif key == "epochs":
            params.epochs = value
        elif key == "learn_rate":
            params.learn_rate = value
        elif key == "losses_to_use":
            params.losses_to_use = value
        elif key == "number_of_data_points":
            params.number_of_data_points = value
        elif key == "percentage_for_validation":
            params.percentage_for_validation = value
        elif key == "batch_size":
            params.batch_size = value
        else:
            raise ValueError(f"The key {key} is not a training parameter")
    return params


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
            with open(
                os.path.join(directory_path, net_name + "_nn_params.json"), "w"
            ) as json_file:
                json.dump(nn.nn_params_dataclass_to_dict(net.params), json_file)

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


class nn_factory(ABC):
    def __init__(self, params):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.batch_size = params.batch_size

        self.data_loss = torch.nn.MSELoss()
        self.PDE_loss_type = torch.nn.MSELoss()

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

        self.epochs = params.epochs
        self.dataless = "data" not in params.losses_to_use
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

    def get_loader(self, data: list, shuffle: bool = False) -> DataLoader:
        if self.dataless:
            return DataLoader(
                dataset=TensorDataset(torch.tensor(data[0])),
                batch_size=self.batch_size,
                shuffle=shuffle,
            )
        else:
            return DataLoader(
                dataset=TensorDataset(torch.tensor(data[0]), torch.tensor(data[1])),
                batch_size=self.batch_size,
                shuffle=shuffle,
            )

    def fit(
        self,
        training_data: list,
        validation_data: list = None,
        verbose: bool = False,
        shuffle_data: bool = False,
        save_losses: bool = True,
        output_dir: str = None,
    ):
        training_loader = self.get_loader(training_data, shuffle=shuffle_data)

        if validation_data != None:
            validation_loader = self.get_loader(training_data, shuffle=shuffle_data)
            track_validation = True
        else:
            validation_loader = None
            track_validation = False

        losses = []
        for i in range(self.epochs):
            training_loss, validation_loss = self.one_grad_descent_iter(
                training_loader,
                validation_loader=validation_loader,
                track_validation=track_validation,
            )

            losses.append([training_loss, validation_loss])

            if verbose:
                print(f"epoch = {i+1}")
                print(f"training loss = {training_loss}")
                print(f"validation loss = {validation_loss}\n")
        if save_losses:
            pd.DataFrame(
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
            ).to_csv(os.path.join(output_dir, "losses.csv"), index=False)

        return self.get_nn_solver()

    def one_grad_descent_iter(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader = None,
        track_validation: bool = False,
    ) -> list:
        training_loss = [[], [], []]
        if track_validation:
            validation_loss = [[], [], []]
        else:
            validation_loss = [[float("Nan")], [float("Nan")], [float("Nan")]]

        self.train()
        for batch in training_loader:
            self.zero_grad()
            losses = self.losses(batch)
            total_loss = sum(losses)
            total_loss.backward()
            self.step()
            training_loss[0].append(total_loss.item())
            training_loss[1].append(losses[0].item())
            training_loss[2].append(losses[1].item())

        if track_validation:
            for batch in validation_loader:
                losses = self.losses(batch)
                total_loss = sum(losses)

                validation_loss[0].append(total_loss.item())
                validation_loss[1].append(losses[0].item())
                validation_loss[2].append(losses[1].item())

        return [np.array(loss).mean(axis=0) for loss in training_loss], [
            np.array(loss).mean(axis=0) for loss in validation_loss
        ]


sys.path.append("./src/AI/PDEs/")
import Darcy_trainers as Dt


def get_nn_factory(params: training_params) -> nn_factory:
    if params.formulation_params.PDE == "Darcy_primal":
        return Dt.Darcy_primal_nn_factory(params)
    elif params.formulation_params.PDE == "Darcy_dual":
        return Dt.Darcy_dual_nn_factory(params)
    else:
        ValueError(f"The PDE {params.PDE} is not implemented")
