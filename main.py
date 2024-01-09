import argparse
import json
import os
import sys
import torch

sys.path.append("./src/")
import engine as ng


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)

    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_json", type=str, help="path_to_json", default="./")
    args = parser.parse_args()
    with open(args.path_to_json, "r") as json_file:
        json_file = json.load(json_file)

    output_dir = os.path.join(
        os.path.dirname(args.path_to_json), json_file["output_dir"]
    )

    nn_solver = ng.do_train(
        json_file["formulation_params"],
        json_file["training_params"],
        output_dir,
        verbose=True,
    )

    # solver = Darcy_Solver().load(os.path.join(output_dir, "nets"))
