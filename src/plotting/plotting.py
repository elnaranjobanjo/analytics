import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression as corre
import torch


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


def make_hp_search_summary_plots(
    output_dir: str, evals: torch.tensor, test_gt: torch.tensor
) -> None:
    losses = []
    max_epochs = 0
    for dir in os.listdir(os.path.join(output_dir, "trials")):
        add = os.path.join(output_dir, "trials", dir)
        if os.path.exists(os.path.join(add, "losses.csv")):
            make_loss_plots(add)
            losses.append(pd.read_csv(os.path.join(add, "losses.csv")))
            max_epochs = max(max_epochs, len(losses[-1]))

    best = pd.read_csv(os.path.join(output_dir, "best_trial", "losses.csv"))

    # compute summary stats
    mse_loss = torch.nn.MSELoss()
    r2 = r2_score(test_gt.numpy(), evals.numpy())
    mse = mse_loss(test_gt, evals)
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
        plt.text(
            0.05,
            0.95,
            f"test r2: {r2}, mse = {mse}",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
        )
        plt.savefig(os.path.join(output_dir, "summary_" + title + ".png"))
        plt.close()
