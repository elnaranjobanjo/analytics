from dataclasses import dataclass
import ray
import sys


sys.path.append("./src/AI/")

import neural_networks as nn


@dataclass
class hp_search_params:
    type: str = "Bayesian"
    nn_type: ray.tune.choice = ray.tune.choice(["dense_nn"])
    hidden_size: ray.tune.choice = ray.tune.choice([32, 64, 128])
    activations: ray.tune.choice = ray.tune.choice(
        [
            "GeLU",
            "ReLU",
            "leakyReLU",
            "tanh",
            "sigmoid",
            "softplus",
            "ELU",
        ]
    )


def print_hp_search_params(params: hp_search_params) -> None:
    print(f"hp search specs:")
    print(f"type =  {params.type}")
    print(f"")


def make_hp_search_params_dataclass(params_dict: dict) -> hp_search_params:
    params = hp_search_params()
    for key, value in params.items():
        if key == "type":
            params.type = value
        else:
            ValueError(f"The parameter {key} is not a valid hp search param")
    return params


class hp_search:
    def __init__(self, params: hp_search_params):
        pass

    def fit(self, data):
        pass
