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
sys.path.append("./src/trainers/")

import FEM_solver as S
import trainer as T


def do_train(PDE: str, params_dict: dict, output_dir: str, verbose=False):
    params = make_training_params_dataclass(PDE, params_dict)
    print("Training nets begins with the following parameters\n")
    print_training_params(params)
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

        FEM_solver = S.Darcy_FEM_Solver(params)
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


def make_training_params_dataclass(PDE: str, params_dict: dict) -> T.training_params:
    params = T.training_params(PDE=PDE)
    params.PDE = PDE
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
        elif key == "losses_to_use":
            params.losses_to_use = value
        elif key == "number_of_data_points":
            params.number_of_data_points = value
        elif key == "percentage_for_validation":
            params.percentage_for_validation = value
        elif key == "batch_size":
            params.batch_size = value
        else:
            raise ValueError(f"The key {key} is not a Darcy training parameter")
    return params


def print_training_params(params: T.training_params) -> None:
    print(f"PDE =  {params.PDE}")
    print(f"degree = {params.degree}")
    print(f"f = {params.f}")
    print(f"A matrix params = {params.A_matrix_params}")
    print(f"epochs = {params.epochs}")
    print(f"learn rate = {params.learn_rate}")
    print(f"losses_to_use = {params.losses_to_use}")
    print(f"number of data points = {params.number_of_data_points}")
    print(f"percentage set for validation = {params.percentage_for_validation}")
    print(f"batch size = {params.batch_size}\n")
