import contextlib
from dataclasses import dataclass, field
import sys
from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter
import os

sys.path.append("./src/formulations/")
sys.path.append("./src/AI/")

import formulation as F
import neural_networks as nn
import trainer as T


@dataclass
class hp_search_params:
    type: str = "Bayesian"
    max_concurrent: int = 1
    num_samples: int = 50
    time_budget_min: int = 30
    scheduler: str = "MedianStoppingRule"
    nn_types: list = field(default_factory=lambda: ["dense_nn"])
    hidden_size_multipliers: list = field(default_factory=lambda: [2 / 3])
    num_layers: int = field(default_factory=lambda: [6])
    activations: list = field(default_factory=lambda: ["GeLU"])


def print_hp_search_params(params: hp_search_params) -> None:
    print(f"hp search specs:")
    print(f"type =  {params.type}")
    print(f"max_concurrent = {params.max_concurrent}")
    print(f"num_samples = {params.num_samples}")
    print(f"time_budget_min = {params.time_budget_min}")
    print(f"scheduler = {params.scheduler}")
    print(f"nn_types = {params.nn_types}")
    print(f"hidden_size_multipliers = {params.hidden_size_multipliers}")
    print(f"num_layers = {params.num_layers}")
    print(f"activations = {params.activations}\n")


def make_hp_search_params_dataclass(params_dict: dict) -> hp_search_params:
    params = hp_search_params()
    for key, value in params_dict.items():
        if key == "type":
            params.type = value
        elif key == "max_concurrent":
            params.max_concurrent = value
        elif key == "num_samples":
            params.num_samples = value
        elif key == "time_budget_min":
            params.time_budget_min = value
        elif key == "scheduler":
            params.scheduler = value
        elif key == "nn_types":
            params.nn_types = value
        elif key == "hidden_size_multipliers":
            params.hidden_size_multipliers = value
        elif key == "num_layers":
            params.num_layers = value
        elif key == "activations":
            params.activations = value
        else:
            raise ValueError(f"The parameter {key} is not a valid hp search param")
    return params


def get_search_algorithm(params: hp_search_params):
    if params.type == "Bayesian":
        return BayesOptSearch()
    else:
        raise ValueError(f"The search algorithm {params.type} is not implemented.")


def get_scheduler(params: hp_search_params):
    if params.scheduler == "MedianStoppingRule":
        return tune.schedulers.MedianStoppingRule()
    elif params.scheduler == "ASHAScheduler":
        return tune.schedulers.ASHAScheduler()
    else:
        raise ValueError(f"The search algorithm {params.type} is not implemented.")


@contextlib.contextmanager
def redirect_stdout_stderr_to_file(file_path):
    with open(file_path, "a") as f:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = f, f

        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def run_optimization(
    formulation_params: F.formulation_params,
    hp_params: hp_search_params,
    training_data: list,
    validation_data: list,
    output_dir: str,
) -> tune.ResultGrid:
    search_space = {
        "type": tune.choice(hp_params.nn_types),
        "hidden_size_multipliers": tune.choice(hp_params.hidden_size_multipliers),
        "num_layers": tune.choice(hp_params.num_layers),
        "activation": tune.choice(hp_params.activations),
    }

    def objective(config):
        nn_params = nn.make_nn_params_dataclass(config)
        nn_factory = T.get_nn_factory(
            formulation_params, nn_params, T.training_params()
        )
        nn_solver, t_loss, v_loss = nn_factory.fit(
            training_data, validation_data=validation_data, save_losses=False
        )
        train.report({"score": v_loss})

    with redirect_stdout_stderr_to_file(
        os.path.join(output_dir, "ray_tune_output.txt")
    ):
        tuner = tune.Tuner(
            objective,
            tune_config=tune.TuneConfig(
                metric="score",
                mode="min",
                search_alg=get_search_algorithm(hp_params),
                num_samples=hp_params.num_samples,
                scheduler=get_scheduler(hp_params),
                time_budget_s=hp_params.time_budget_min * 60,
            ),
            param_space=search_space,
        )
        tuner = ConcurrencyLimiter(tuner, max_concurrent=hp_params.max_concurrent)
        return tuner.fit()
