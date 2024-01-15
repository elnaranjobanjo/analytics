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
    parser.add_argument("-tasks", type=str, help="tasks available: train, hp_tune")
    args = parser.parse_args()
    with open(args.path_to_json, "r") as json_file:
        json_file = json.load(json_file)

    output_dir = os.path.join(
        os.path.dirname(args.path_to_json), json_file["output_dir"]
    )
    if args.tasks == "train":
        nn_solver = ng.do_train(
            json_file["data_gen_params"],
            json_file["formulation_params"],
            json_file["nn_params"],
            json_file["training_params"],
            output_dir,
            verbose=True,
        )
    elif args.tasks == "hp_search":
        ng.do_hp_tuning(
            json_file["data_gen_params"],
            json_file["formulation_params"],
            json_file["hp_search"],
            output_dir,
            verbose=True,
        )
    else:
        raise ValueError(f"The task {args.tasks} is not implemented.")
