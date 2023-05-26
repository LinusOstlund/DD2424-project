import torch
import torch.nn as nn
import torch.utils.data as tud
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# used to avoid clustering the repo with checkpoints
import tempfile


from lstm_model import *
from transf_model import *
from dataset import *
import train
import inference
import mlflow
import subprocess


def convert_abc_to_midi(run_id) -> None:
    """
    Convert all the .abc files in inference/ to .mid files in midi/
    Then store them as artifacts in MLflow
    """

    abc_dir = "inference/"
    midi_dir = "midi/"

    if not os.path.exists(midi_dir):
        os.makedirs(midi_dir)

    if not os.path.exists(abc_dir):
        os.makedirs(abc_dir)

    # get the run
    run = mlflow.get_run(run_id)

    artifact_uri = "runs:/" + run_id + "/inference/"

    mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri, dst_path="inference/"
    )

    # create midi_dir if it doesn't exist
    os.makedirs(midi_dir, exist_ok=True)

    # for each file in abc_dir
    for file in os.listdir(abc_dir):
        # base is the file name without the extension
        base = file.split(".")[0]
        input_file = os.path.join(abc_dir, file)
        output_file = os.path.join(midi_dir, base + ".mid")

        print(f"Processing {input_file}")
        subprocess.run(["abc2midi", input_file, "-o", output_file])

    # log all the midi files as artifacts
    mlflow.log_artifacts(midi_dir, artifact_path="midi")


def main(args) -> None:
    """
    This is a script wrapper for training a model.
    It is called from the MLproject file.
    """
    active_run = mlflow.start_run()
    # fetch run id
    run_id = active_run.info.run_id

    # for arg in vars(args):
    #    mlflow.log_param(arg, getattr(args, arg))

    # Step 1: train
    train.main(args)

    # Step 2: inference (generate samples)
    args.run_id = run_id
    inference.main(args)

    # Step 3: evaluate (convert to midi)
    if args.convert_to_midi:
        convert_abc_to_midi(run_id)

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for training
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-hs", "--hidden_size", type=int, default=16)
    parser.add_argument("-es", "--embedding_size", type=int, default=16)
    parser.add_argument("-l", "--layers", type=int, default=2)
    parser.add_argument("-dp", "--dropout", type=float, default=0.2)
    parser.add_argument("-ah", "--attention_heads", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.9)
    parser.add_argument("-p", "--patience", type=int, default=5)
    parser.add_argument("-ld", "--load", type=str, default=None)  # TODO implement
    parser.add_argument("-ml", "--max_sequence_length", type=int, default=256)
    parser.add_argument("-arch", "--architecture", type=str, default="lstm")
    parser.add_argument("-eo", "--epochs_offset", type=int, default=0)
    parser.add_argument("-ne", "--new_experiment", type=str, default=None)
    parser.add_argument("-en", "--experiment_name", type=str, default=None)
    parser.add_argument("-tu", "--tracking_uri", type=str)
    parser.add_argument("-rs", "--random_seed", type=int, default=0)

    # Arguments for inference
    parser.add_argument("-vcb", "--vocab_file", type=str, default=None)
    parser.add_argument("-m", "--mode", type=str, default="topk")
    parser.add_argument("-t", "--temperature", type=float, default=0.6)
    parser.add_argument("-n", "--sample_num", type=int, default=100)

    # abc file parameters for ease of analysis
    parser.add_argument("-d", "--delimeter", type=str, default="$")  # not used
    parser.add_argument("-midi", "--convert_to_midi", type=str, default=None)
    args = parser.parse_args()

    assert args.architecture in [
        "lstm",
        "transf",
    ], "Architecture must be either 'lstm' or 'transf'"

    if args.architecture == "tranf":
        assert (
            args.attention_heads is not None
        ), "Attention heads must be specified for transformer architecture"

    """    
    # set experiment id as an argument
    if args.new_experiment:
        experiment_id = mlflow.create_experiment(args.new_experiment)
    elif args.experiment_name:
        experiment_id = mlflow.get_experiment_by_name(
            args.experiment_name
        ).experiment_id
    else:
        raise ValueError(
            "Either --new_experiment or --experiment_name must be specified"
        )
    """
    # TODO make this configurable (now it just uses the default)
    mlflow.set_tracking_uri(args.tracking_uri)

    main(args)

"""

python src/main.py --epochs 40 \
                    --vocab_file data/vocab.json \
                    --experiment_name "My Experiment" \
                    -hs 128 \
                    -es 64 \
                    --layers 2 \
                    --dropout 0.2 \
                    --learning_rate 0.001 \
                    --batch_size 32 \
                    -midi True
                    

"""
