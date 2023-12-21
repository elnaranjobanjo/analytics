import fenics as fe
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

import Darcy_trainer as Dt
import Darcy_generator as Dg


def do_train(params: Dt.DarcyTrainingParams, output_dir: str, verbose=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(os.path.join(output_dir, "log.txt")):
        os.remove(os.path.join(output_dir, "log.txt"))

    generated_A_matrix_params = [
        [
            random.uniform(params.A_matrix_params[0][0], params.A_matrix_params[0][1]),
            random.uniform(params.A_matrix_params[1][0], params.A_matrix_params[1][1]),
            random.uniform(0, 2 * np.pi),
        ]
        for _ in range(params.number_of_data_points)
    ]
    split_index = int(params.percentage_for_validation * len(generated_A_matrix_params))

    X_train, X_val = (
        generated_A_matrix_params[split_index:],
        generated_A_matrix_params[:split_index],
    )

    train_csv = pd.DataFrame(X_train, columns=["eig_1", "eig_2", "theta"])
    val_csv = pd.DataFrame(X_val, columns=["eig_1", "eig_2", "theta"])

    if params.dataless:
        training_data = [X_train]
        validation_data = [X_val]

    else:
        sim_params = Dg.DarcySimParams(
            mesh=params.mesh, degree=params.degree, f=params.f
        )
        Y = Dg.generate_data_using_eig_rep(sim_params, generated_A_matrix_params)
        Y_train, Y_val = (
            Y[split_index:],
            Y[:split_index],
        )
        training_data = [X_train, Y_train]
        validation_data = [X_val, Y_val]

        Y_train_csv = pd.DataFrame(Y_train)
        Y_val_csv = pd.DataFrame(Y_val)

        train_csv = pd.concat([train_csv, Y_train_csv], axis=1)
        val_csv = pd.concat([val_csv, Y_val_csv], axis=1)

    train_csv.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_csv.to_csv(os.path.join(output_dir, "val.csv"), index=False)

    factory_params = Dt.DarcynnFactoryParams(
        mesh=params.mesh,
        degree=params.degree,
        f=params.f,
        epochs=params.epochs,
        learn_rate=params.learn_rate,
    )

    nn_solver = Dt.Darcy_nn_Factory(factory_params).fit(
        training_data, validation_data, params.batch_size, output_dir, verbose=verbose
    )
    nn_solver.save(os.path.join(output_dir, "nets"))
    make_loss_plot(output_dir)
    return


def make_loss_plot(output_dir: str):
    df = pd.read_csv(os.path.join(output_dir, "loss.csv"))
    training = df["training"].to_numpy()
    validation = df["validation"].to_numpy()

    fig, axes = plt.subplots(1, 2)

    axes[0].plot(range(1, len(training) + 1), training, color="black")
    # axes[0].scatter(range(1, len(training) + 1), training, color="black", marker="o")
    axes[0].set_ylabel("PDE loss")
    axes[0].set_xlabel("epochs")
    axes[0].set_title("training")

    axes[1].plot(
        range(1, len(validation) + 1),
        validation,
    )
    axes[1].plot(range(1, len(validation) + 1), validation, color="black")
    # axes[1].scatter(
    #    range(1, len(validation) + 1), validation, color="black", marker="o"
    # )
    # axes[1].set_ylabel("PDE loss")
    axes[1].set_xlabel("epochs")
    axes[1].set_title("validation")

    fig.suptitle("Darcy Dataless Training", fontsize=16)
    plt.savefig(os.path.join(output_dir, "loss_plots.png"))


def make_DarcyTrainingParams_dataclass(params_dict: dict) -> Dt.DarcyTrainingParams:
    params = Dt.DarcyTrainingParams()
    for key, value in params_dict.items():
        if key == "mesh":
            params.mesh = fe.Mesh(value)
        elif key == "degree":
            params.degree = value
        elif key == "f":
            params.f = value
        elif key == "A_matrix_params":
            params.A_matrix_params = value
        elif key == "epochs":
            params.epochs = value
        elif key == "learn_rate":
            params.learn_rate = value
        elif key == "dataless":
            params.dataless = value
        elif key == "number_of_data_points":
            params.number_of_data_points = value
        elif key == "percentage_for_validation":
            params.percentage_for_validation = value
        elif key == "batch_size":
            params.batch_size = value
        else:
            raise ValueError(f"The key {key} is not a Darcy training parameter")
    return params
