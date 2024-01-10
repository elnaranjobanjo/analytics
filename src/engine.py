import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import sys
import time

sys.path.append("./src/FEM_solvers/")
sys.path.append("./src/formulations/")
sys.path.append("./src/hp_tuning/")
sys.path.append("./src/AI/")

import FEM_solver as S
import formulation as F
import hp_tuning as H
import neural_networks as nn
import trainer as T


def do_train(
    formulation_dict: dict, nn_dict, training_dict: dict, output_dir: str, verbose=False
):
    params = T.make_training_params_dataclass(formulation_dict, nn_dict, training_dict)
    print("Training nets begins with the following parameters\n")
    T.print_training_params(params)
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

    if "data" in params.losses_to_use:
        print("Generating Training Data\n")
        with open(os.path.join(output_dir, "log.txt"), "a") as file:
            file.write(f"Generating training data")

        FEM_solver = S.Darcy_FEM_Solver(params.formulation_params)
        time_1 = time.time()
        Y = np.array(
            list(
                map(
                    lambda x: FEM_solver.compute_solution_eig_rep(x),
                    generated_A_matrix_params,
                )
            )
        )
        time_2 = time.time()
        with open(os.path.join(output_dir, "log.txt"), "a") as file:
            file.write(
                f"finished! it took {time_2-time_1} seconds or {(time_2-time_1)/3600}"
            )
        Y_train, Y_val = (
            Y[split_index:],
            Y[:split_index],
        )
        training_data = [X_train, Y_train]
        validation_data = [X_val, Y_val]

        # num_u_dofs = FEM_solver.formulation.get_model_space().sub(0).dim()
        # num_p_dofs = FEM_solver.formulation.get_model_space().sub(1).dim()
        # column_labels = [f"u_{i}" for i in range(num_u_dofs)] + [
        #     f"p_{i}" for i in range(num_p_dofs)
        # ]
        num_p_dofs = FEM_solver.formulation.get_model_space().dim()
        column_labels = [f"p_{i}" for i in range(num_p_dofs)]

        Y_train_csv = pd.DataFrame(Y_train, columns=column_labels)
        Y_val_csv = pd.DataFrame(Y_val, columns=column_labels)

        train_csv = pd.concat([train_csv, Y_train_csv], axis=1)
        val_csv = pd.concat([val_csv, Y_val_csv], axis=1)
    else:
        training_data = [X_train]
        validation_data = [X_val]

    train_csv.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_csv.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    print("Training nets\n")
    nn_solver = T.nn_Factory(params).fit(
        training_data, validation_data, params.batch_size, output_dir, verbose=verbose
    )
    nn_solver.save(os.path.join(output_dir, "nets"))
    make_loss_plots(output_dir)
    return nn_solver


def do_hp_tuning(hp_dict: dict, training_dict: dict, output_dir: str, verbose=False):
    pass


def make_loss_plots(output_dir: str) -> None:
    df = pd.read_csv(os.path.join(output_dir, "loss.csv"))
    for title in df.columns:
        y = df[title].to_numpy()
        plt.plot(range(1, len(y) + 1), y, color="black")
        plt.title(title)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(os.path.join(output_dir, title + ".png"))
        plt.close()
