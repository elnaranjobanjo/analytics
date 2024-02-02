import contextlib
from dataclasses import dataclass, field
from ray import train, tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
import os
import torch
import sys

import src.formulations.formulation as F
import src.AI.neural_networks as nn
import src.AI.nn_factory as nn_F


@dataclass
class training_params_search_specs:
    epochs: list = field(default_factory=lambda: [50])
    learn_rate: list = field(default_factory=lambda: [0.001])
    batch_size: list = field(default_factory=lambda: [100])


def print_training_params_search_specs(params: training_params_search_specs) -> None:
    print(f"training params search specs:")
    print(f"epochs = {params.epochs}")
    print(f"learn_rate = {params.learn_rate}")
    print(f"batch_size = {params.batch_size}\n")


def make_training_params_search_dataclass(
    params_dict: dict,
) -> training_params_search_specs:
    params = training_params_search_specs()
    for key, value in params_dict.items():
        if key == "epochs":
            params.epochs = value
        elif key == "learn_rate":
            params.learn_rate = value
        elif key == "batch_size":
            params.batch_size = value
        else:
            raise ValueError(
                f"The parameter {key} is not a valid training param search param."
            )
    return params


@dataclass
class nn_architecture_search_specs:
    nn_types: list = field(default_factory=lambda: ["dense_nn"])
    hidden_size_multipliers: list = field(default_factory=lambda: [2 / 3])
    num_layers: int = field(default_factory=lambda: [6])
    activations: list = field(default_factory=lambda: ["GeLU"])


def print_nn_architecture_search_specs(params: nn_architecture_search_specs) -> None:
    print(f"nn architecture search specs:")
    print(f"nn_types = {params.nn_types}")
    print(f"hidden_size_multipliers = {params.hidden_size_multipliers}")
    print(f"num_layers = {params.num_layers}")
    print(f"activations = {params.activations}\n")


def make_nn_architecture_dataclass(params_dict: dict) -> nn_architecture_search_specs:
    params = nn_architecture_search_specs()
    for key, value in params_dict.items():
        if key == "nn_types":
            params.nn_types = value
        elif key == "hidden_size_multipliers":
            params.hidden_size_multipliers = value
        elif key == "num_layers":
            params.num_layers = value
        elif key == "activations":
            params.activations = value
        else:
            raise ValueError(
                f"The parameter {key} is not a valid nn architecture search param."
            )
    return params


@dataclass
class hp_search_params:
    type: str = "HyperOptSearch"
    max_concurrent: int = 1
    num_samples: int = 50
    time_budget_hrs: int = 0.5
    scheduler: str = "MedianStoppingRule"
    training_params_search_space: training_params_search_specs = (
        training_params_search_specs()
    )
    nn_architecture_search_space: nn_architecture_search_space = (
        nn_architecture_search_specs()
    )


def print_hp_search_params(params: hp_search_params) -> None:
    print(f"hp search specs:")
    print(f"type =  {params.type}")
    print(f"max_concurrent = {params.max_concurrent}")
    print(f"num_samples = {params.num_samples}")
    print(f"time_budget_hrs = {params.time_budget_hrs}")
    print(f"scheduler = {params.scheduler}\n")
    print_training_params_search_specs(params.training_params_search_space)
    print_nn_architecture_search_specs(params.nn_architecture_search_space)


def make_hp_search_params_dataclass(params_dict: dict) -> hp_search_params:
    params = hp_search_params()
    for key, value in params_dict.items():
        if key == "type":
            params.type = value
        elif key == "max_concurrent":
            params.max_concurrent = value
        elif key == "num_samples":
            params.num_samples = value
        elif key == "time_budget_hrs":
            params.time_budget_hrs = value
        elif key == "scheduler":
            params.scheduler = value
        elif key == "training_params":
            params.training_params_search_space = make_training_params_search_dataclass(
                value
            )
        elif key == "nn_architecture":
            params.nn_architecture_search_space = make_nn_architecture_dataclass(value)
        else:
            raise ValueError(f"The parameter {key} is not a valid hp search param.")
    return params


def get_search_algorithm(params: hp_search_params):
    if params.type == "HyperOptSearch":
        return HyperOptSearch()
    else:
        raise ValueError(f"The search algorithm {params.type} is not implemented.")


def get_scheduler(params: hp_search_params):
    if params.scheduler == "MedianStoppingRule":
        return tune.schedulers.MedianStoppingRule()
    elif params.scheduler == "ASHAScheduler":
        return tune.schedulers.ASHAScheduler()
    else:
        raise ValueError(f"The search algorithm {params.type} is not implemented.")


def define_search_space(params: hp_search_params) -> dict:
    # return {"training_params": {}, "nn_architecture": {}}
    return {
        "batch_size": tune.choice(
            range(
                params.training_params_search_space.batch_size[0],
                params.training_params_search_space.batch_size[1],
            )
        ),
        "learn_rate": tune.loguniform(
            params.training_params_search_space.learn_rate[0],
            params.training_params_search_space.learn_rate[1],
        ),
        "epochs": tune.choice(
            range(
                params.training_params_search_space.epochs[0],
                params.training_params_search_space.epochs[1],
            )
        ),
        "type": tune.choice(params.nn_architecture_search_space.nn_types),
        "hidden_size_multiplier": tune.uniform(
            params.nn_architecture_search_space.hidden_size_multipliers[0],
            params.nn_architecture_search_space.hidden_size_multipliers[1],
        ),
        "num_layers": tune.choice(
            range(
                params.nn_architecture_search_space.num_layers[0],
                params.nn_architecture_search_space.num_layers[1],
            )
        ),
        "activation": tune.choice(params.nn_architecture_search_space.activations),
    }


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
    training_params: nn_F.training_params,
    hp_params: hp_search_params,
    training_data: list,
    validation_data: list,
    output_dir: str,
    verbose: bool = False,
) -> tune.ResultGrid:
    def objective(config):
        torch.set_default_dtype(torch.double)
        t_params = nn_F.make_training_params_dataclass(
            config, replace_from=training_params, raise_err=False
        )
        nn_params = nn.make_nn_params_dataclass(config, raise_err=False)
        nn_factory = nn_F.get_nn_factory(formulation_params, nn_params, t_params)
        trial_dir = train.get_context().get_trial_dir()
        nn_solver, t_loss, v_loss = nn_factory.fit(
            training_data,
            validation_data=validation_data,
            output_dir=trial_dir,
            save_losses=True,
        )
        nn_solver.save(os.path.join(trial_dir, "nets"))
        train.report({"score": v_loss})

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            search_alg=ConcurrencyLimiter(
                get_search_algorithm(hp_params),
                max_concurrent=int(hp_params.max_concurrent),
            ),
            num_samples=hp_params.num_samples,
            scheduler=get_scheduler(hp_params),
            time_budget_s=hp_params.time_budget_hrs * 60 * 60,
        ),
        run_config=train.RunConfig(
            local_dir=output_dir,
            # storage_path=output_dir,
            name="trials",
            log_to_file=False,
        ),
        param_space=define_search_space(hp_params),
    )
    if verbose:
        return tuner.fit()
    else:
        with redirect_stdout_stderr_to_file(
            os.path.join(output_dir, "ray_tune_output.txt")
        ):
            return tuner.fit()
