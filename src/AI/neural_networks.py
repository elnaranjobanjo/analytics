from dataclasses import dataclass, field
import torch


@dataclass
class nn_params:
    type: str = "dense_nn"
    hidden_size_multiplier: float = 2 / 3
    num_layers: int = 5
    activation: str = "GeLU"
    dropout_rate: float = 0.0


def print_neural_net_params(params: nn_params) -> None:
    print(f"Neural net architecture specs:")
    print(f"type = {params.type}")
    print(f"hidden_size_multiplier = {params.hidden_size_multiplier}")
    print(f"num_layers = {params.num_layers}")
    print(f"activation = {params.activation}")
    print(f"dropout_rate = {params.dropout_rate}\n")


def make_nn_params_dataclass(params_dict: dict, raise_err=True) -> nn_params:
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
        elif key == "dropout_rate":
            params.dropout_rate = value
        else:
            if raise_err:
                raise ValueError(f"The entry {key} is not a valid nn parameter")
    return params


def nn_params_dataclass_to_dict(dataclass: nn_params) -> dict:
    return {
        "type": dataclass.type,
        "hidden_size_multiplier": dataclass.hidden_size_multiplier,
        "num_layers": dataclass.num_layers,
        "activation": dataclass.activation,
        "dropout_rate": dataclass.dropout_rate,
    }


def dense_net_size_heuristic(
    input_size: int, output_size: int, multiplier: float
) -> int:
    return int(multiplier * (input_size + output_size))


def get_activation_func(params: nn_params) -> torch.nn:
    if params.activation == "GeLU":
        return torch.nn.GELU()
    elif params.activation == "ReLU":
        return torch.nn.ReLU()
    elif params.activation == "LeakyReLU":
        return torch.nn.LeakyReLU()
    elif params.activation == "Tanh":
        return torch.nn.Tanh()
    elif params.activation == "Sigmoid":
        return torch.nn.Sigmoid()
    elif params.activation == "Softplus":
        return torch.nn.Softplus()
    elif params.activation == "ELU":
        return torch.nn.ELU()
    else:
        raise ValueError(f"The activation {params.activation} is not implemented")


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

        self.activation = get_activation_func(params)

        hidden_size = dense_net_size_heuristic(
            input_size, output_size, params.hidden_size_multiplier
        )

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))

        for _ in range(params.num_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            # if params.dropout_rate > 0:
            #     self.layers.append(torch.nn.Dropout(p=params.dropout_rate))

        self.layers.append(torch.nn.Linear(hidden_size, output_size))

        self.params = params

    def forward(self, x: torch.tensor) -> torch.tensor:
        # for layer in self.layers[:-1]:
        #     x = layer(x)
        #     if isinstance(layer, torch.nn.Linear):
        #         x = self.activation(x)
        # return self.layers[-1](x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
