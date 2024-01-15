from dataclasses import dataclass, field
import torch


@dataclass
class nn_params:
    type: str = "dense_nn"
    hidden_size_multiplier: float = 2 / 3
    num_layers: int = 5
    activation: str = "GeLU"


def print_neural_net_params(params: nn_params) -> None:
    print(f"Neural net architecture specs:")
    print(f"type = {params.type}")
    print(f"hidden_size_multiplier = {params.hidden_size_multiplier}")
    print(f"num_layers = {params.num_layers}")
    print(f"activation = {params.activation}\n")


def make_nn_params_dataclass(params_dict: dict) -> nn_params:
    params = nn_params()
    for key, value in params_dict.items():
        if key == "type":
            params.type = value
        elif key == "hidden_size_multiplier":
            params.hidden_size_multiplier = value
        elif key == "num_layers":
            params.num_layers == value
        elif key == "activation":
            params.activation = value
        else:
            raise ValueError(f"The entry {key} is not a valid nn parameter")
    return params


def nn_params_dataclass_to_dict(dataclass: nn_params) -> dict:
    return {
        "type": dataclass.type,
        "hidden_size_multiplier": dataclass.hidden_size_multiplier,
        "num_layers": dataclass.num_layers,
        "activation": dataclass.activation,
    }


def dense_net_size_heuristic(
    input_size: int, output_size: int, multiplier: float
) -> int:
    return int(multiplier * (input_size + output_size))


def initialize_nn(input_size: int, output_size: int, params: nn_params):
    if params.type == "dense_nn":
        return dense_nn(input_size, output_size, params)
    else:
        raise ValueError(f"The neural net achitecture {params.type} is not implemented")


class dense_nn(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, params: nn_params):
        super(dense_nn, self).__init__()
        if params.num_layers < 2:
            raise ValueError("Number of layers must be at least 2")

        if params.activation == "GeLU":
            self.activation = torch.nn.GELU()
        elif params.activation == "ReLU":
            self.activation = torch.nn.ReLU()
        elif params.activation == "LeakyReLU":
            self.activation = torch.nn.LeakyReLU()
        elif params.activation == "Tanh":
            self.activation = torch.nn.Tanh()
        elif params.activation == "Sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif params.activation == "Softplus":
            self.activation = torch.nn.Softplus()
        elif params.activation == "ELU":
            self.activation = torch.nn.ELU()
        else:
            raise ValueError(f"The activation {params.activation} is not implemented")

        hidden_size = dense_net_size_heuristic(
            input_size, output_size, params.hidden_size_multiplier
        )

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))

        for _ in range(params.num_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))

        self.layers.append(torch.nn.Linear(hidden_size, output_size))

        self.params = params

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
