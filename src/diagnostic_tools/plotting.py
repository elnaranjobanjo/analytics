import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import r_regression as corre
from typing import Tuple
import torch

import src.formulations.formulation as F


def make_loss_plots(output_dir: str) -> None:
    df = pd.read_csv(os.path.join(output_dir, "losses.csv"))
    for title in df.columns:
        plt.plot(
            range(1, len(df[title].to_numpy()) + 1), df[title].to_numpy(), color="black"
        )
        plt.title(title)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(os.path.join(output_dir, title + ".png"))
        plt.close()


def make_hp_search_summary_plots(output_dir: str) -> None:
    losses = []
    max_epochs = 0
    for dir in os.listdir(os.path.join(output_dir, "trials")):
        add = os.path.join(output_dir, "trials", dir)
        if os.path.exists(os.path.join(add, "losses.csv")):
            make_loss_plots(add)
            losses.append(pd.read_csv(os.path.join(add, "losses.csv")))
            max_epochs = max(max_epochs, len(losses[-1]))

    best = pd.read_csv(os.path.join(output_dir, "best_trial", "losses.csv"))

    for title in losses[0].columns:
        for loss in losses:
            plt.plot(
                range(1, len(loss[title].to_numpy()) + 1),
                loss[title].to_numpy(),
                color="grey",
                linewidth=2.0,
            )

        plt.plot(
            range(1, len(best[title].to_numpy()) + 1),
            best[title].to_numpy(),
            color="black",
        )
        plt.title(title)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        # plt.text(
        #     0.05,
        #     0.95,
        #     f"test r2: {r2}, mse = {mse}",
        #     ha="right",
        #     va="top",
        #     transform=plt.gca().transAxes,
        # )
        plt.savefig(os.path.join(output_dir, "summary_" + title + ".png"))
        plt.close()


def extract_gt_and_evals(
    gt_address: str, evals_address: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(gt_address)
    gt = gt.drop(gt.columns[[0, 1, 2]], axis=1)

    evals = pd.read_csv(evals_address)
    evals = evals.drop(evals.columns[[0, 1, 2]], axis=1)
    return gt, evals


def run_diagnostic_stats(
    gt_address: str,
    evals_address: str,
) -> pd.DataFrame:
    gt, evals = extract_gt_and_evals(gt_address, evals_address)
    summary = pd.DataFrame()
    mse_loss = torch.nn.MSELoss()

    summary["r2"] = [r2_score(gt, evals)]
    summary["mse"] = [
        mse_loss(torch.tensor(gt.to_numpy()), torch.tensor(evals.to_numpy())).item()
    ]

    return summary


def make_data_parity_plots(
    gt_address: str,
    evals_address: str,
    working_dir: str,
) -> None:

    dir = os.path.join(working_dir, "data_pplots")
    if not os.path.exists(dir):
        os.makedirs(dir)

    gt, evals = extract_gt_and_evals(gt_address, evals_address)
    for title in gt.columns:
        x_values = gt[title].to_numpy()
        y_values = evals[title].to_numpy()
        plt.scatter(x_values, y_values)
        plt.plot(x_values, x_values, linestyle="--", color="red")

        plt.title(f"data pplot {title}")
        plt.xlabel("gt")
        plt.ylabel("preds")
        plt.text(
            0.05,
            0.95,
            f"r2 = {r2_score(x_values,y_values):.3g}\nmse = {mean_squared_error(x_values,y_values):.3g}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.savefig(os.path.join(dir, f"{title}.png"))
        plt.close()


def make_PDE_parity_plots(
    gt_f: np.array,
    eval_f: np.array,
    working_dir: str,
) -> None:
    dir = os.path.join(working_dir, "PDE_pplots")
    if not os.path.exists(dir):
        os.makedirs(dir)

    cmap = plt.cm.get_cmap("viridis", len(gt_f))
    colors = [cmap(i) for i in range(len(gt_f))]

    fig1, ax1 = plt.subplots()
    for i, f_i in enumerate(gt_f):
        num_f = eval_f[:, i]
        residuals = num_f - f_i
        num_bins = max(int(len(num_f) / 10), 10)

        hist, bin_edges = np.histogram(residuals, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax1.plot(bin_centers, hist, linestyle="-", linewidth=2, color=colors[i])

        fig2, ax2 = plt.subplots()
        ax2.hist(
            num_f,
            bins=max(int(len(num_f) / 10), 10),
            alpha=0.75,
            color="blue",
        )
        ax2.set_title(f"Histogram f_{i}")
        ax2.set_xlabel(f"predicted f_{i}")
        ax2.set_ylabel("frequency")
        ax2.axvline(x=f_i, color="red", linestyle="--", linewidth=2, label="gt")
        ax2.axvline(
            x=np.mean(num_f),
            color="green",
            linestyle="--",
            linewidth=2,
            label="mean of preds",
        )
        ax2.text(
            0.05,
            0.95,
            f"mean = {np.mean(num_f):.3g}\nstd = {np.std(num_f):.3g}\nexact_f = {f_i:.3g}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="left",
        )
        fig2.legend()
        fig2.savefig(os.path.join(dir, f"f_{i}.png"))
        plt.close(fig2)

    ax1.plot(bin_centers, hist, linestyle="-", linewidth=2, color="blue")
    ax1.set_title("Summary Histogram")
    ax1.set_xlabel("residuals")
    fig1.savefig(os.path.join(working_dir, f"summary_rhs_residuals.png"))
    plt.close(fig1)


def diagnose_linear_system(
    PDE: F.PDE_formulation, A_matrix_params: np.array, title: str, diagnostics_dir: str
) -> None:
    condition_nums = np.array(
        list(
            map(
                lambda x: np.linalg.cond(PDE.assemble_linear_system(x)), A_matrix_params
            )
        )
    )
    plt.hist(
        condition_nums,
        bins=max(int(len(condition_nums) / 10), 10),
        alpha=0.75,
        color="blue",
    )
    plt.text(
        0.05,
        0.95,
        f"mean = {np.mean(condition_nums):.3g}\nstd = {np.std(condition_nums):.3g}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    plt.savefig(os.path.join(diagnostics_dir, f"{title}_condition_nums.png"))
    plt.close()

    pd.DataFrame(
        np.hstack((A_matrix_params, condition_nums.reshape(-1, 1))),
        columns=["eig_1", "eig_2", "theta", "condition_number"],
    ).to_csv(os.path.join(diagnostics_dir, f"{title}_condition_nums.csv"), index=False)


# def make_PDE_parity_plots(
#     gt_f: np.array,
#     eval_f: np.array,
#     working_dir: str,
# ) -> None:
#     scatter_points = []

#     for i in range(len(gt_f)):
#         for j in range(eval_f.shape[0]):
#             scatter_points.append([gt_f[i], eval_f[j, i]])

#     x_values, y_values = zip(*scatter_points)

#     plt.scatter(x_values, y_values)
#     plt.plot(x_values, x_values, linestyle="--", color="red")
#     min_x, max_x, min_y, max_y = (
#         min(x_values),
#         max(x_values),
#         min(y_values),
#         max(y_values),
#     )
#     if min_x < max_x:
#         plt.xlim(min(x_values), max(x_values))
#     if min_y < max_y:
#         plt.ylim(min(y_values), max(y_values))
#     plt.title(f"PDE pplot")
#     plt.xlabel("gt")
#     plt.ylabel("preds")
#     plt.savefig(os.path.join(working_dir, "PDE_pplot.png"))
#     plt.close()
