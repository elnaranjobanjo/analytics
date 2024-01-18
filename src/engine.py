import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

import src.FEM_solvers.FEM_solver as S
import src.hp_tuning.hp_tuning as H
import src.formulations.formulation as F
import src.AI.neural_networks as nn
import src.AI.trainer as T


def do_train(
    data_gen_dict: dict,
    formulation_dict: dict,
    nn_dict: dict,
    training_dict: dict,
    output_dir: str,
    verbose=False,
):
    output_dir = os.path.join(output_dir, "training")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_gen_params = S.make_data_gen_params_dataclass(data_gen_dict)
    formulation_params = F.make_formulation_params_dataclass(formulation_dict)
    nn_params = nn.make_nn_params_dataclass(nn_dict)
    training_params = T.make_training_params_dataclass(training_dict)

    print("Training nets begins with the following parameters\n")
    S.print_data_gen_params(data_gen_params)
    F.print_formulation_params(formulation_params)
    nn.print_neural_net_params(nn_params)
    T.print_training_params(training_params)

    print("Generating Training Data\n")
    training_data, validation_data = S.generate_data(
        data_gen_params,
        formulation_params,
        output_dir=output_dir,
        include_output_vals=("data" in training_params.losses_to_use),
        save_in_csv=True,
    )
    print("Training nets\n")
    nn_factory = T.get_nn_factory(formulation_params, nn_params, training_params)
    nn_solver, t_loss, v_loss = nn_factory.fit(
        training_data,
        validation_data=validation_data,
        verbose=verbose,
        save_losses=True,
        output_dir=output_dir,
    )

    nn_solver.save(os.path.join(output_dir, "nets"))
    make_loss_plots(output_dir)
    return nn_solver


def do_hp_tuning(
    data_gen_dict: dict,
    formulation_dict: dict,
    hp_dict: dict,
    output_dir: str,
    verbose=False,
):
    output_dir = os.path.join(output_dir, "hp_tuning")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    formulation_params = F.make_formulation_params_dataclass(formulation_dict)
    data_gen_params = S.make_data_gen_params_dataclass(data_gen_dict)
    hp_params = H.make_hp_search_params_dataclass(hp_dict)
    print("hyperparameter search begins with the following parameters\n")
    F.print_formulation_params(formulation_params)
    S.print_data_gen_params(data_gen_params)
    H.print_hp_search_params(hp_params)

    print("Generating Training Data\n")
    training_data, validation_data = S.generate_data(
        data_gen_params,
        formulation_params,
        output_dir=output_dir,
        include_output_vals=True,
        save_in_csv=True,
    )

    print("Starting hp search\n")
    try:
        results = H.run_optimization(
            formulation_params,
            hp_params,
            training_data,
            validation_data,
            output_dir,
            verbose=verbose,
        )
        with open(os.path.join(output_dir, "best_hp.json"), "w") as json_file:
            json.dump(results.get_best_result().config, json_file)
        print("Best hyperparameters found were: ", results.get_best_result().config)
    finally:
        # Ray forcibly wants to put a copy of the output here,
        # The only way to avoid it (that I know of) is to just delete it after the fact
        shutil.rmtree(os.path.expanduser(os.path.join("~", "ray_results")))


def make_loss_plots(output_dir: str) -> None:
    df = pd.read_csv(os.path.join(output_dir, "losses.csv"))
    for title in df.columns:
        y = df[title].to_numpy()
        plt.plot(range(1, len(y) + 1), y, color="black")
        plt.title(title)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(os.path.join(output_dir, title + ".png"))
        plt.close()
