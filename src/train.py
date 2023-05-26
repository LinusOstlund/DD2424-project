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
import mlflow


def main(args) -> None:
    torch.manual_seed(args.random_seed)

    # load dataset

    dataset = MusicDataset(
        data_file="data/dataset.json",
        vocab_file="data/vocab.json",
        max_sequence_length=args.max_sequence_length,
    )

    vocab_size = dataset.vocab_size
    B = args.batch_size
    L = dataset.max_sequence_length

    # split data
    train_data, val_data = tud.random_split(
        dataset, [args.train_ratio, 1 - args.train_ratio]
    )

    # TODO gör en split och logga den som artifact?
    #      fast då faller strukturen i MLproject...

    # wrap in data loaders
    train_loader = tud.DataLoader(train_data, batch_size=B, shuffle=True)
    val_loader = tud.DataLoader(val_data, batch_size=len(val_data), shuffle=True)

    # model, optimizer, loss
    if args.architecture == "lstm":
        model = LSTMModel(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            num_layers=args.layers,
            dropout=args.dropout,
            bidirectional=False,
        )
    elif args.architecture == "transf":
        model = TransfModel(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            num_layers=args.layers,
            dropout=args.dropout,
            attention_heads=args.attention_heads,
        )
    else:
        raise ValueError("Invalid architecture, choose lstm or transf.")

    # to keep track of epochs across multiple runs
    offset = args.epochs_offset + 1
    # load previous checkpoint
    if args.load != None:
        # TODO detta ska nog vara kvar ändå? Men nu ligger checkpoints i artifacts
        model.load_state_dict(torch.load(args.load))

    # check gpu
    if torch.cuda.is_available():
        print("Using CUDA.")
        model = model.cuda()
        device = "cuda"

    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        device = torch.device("mps")
        model.to(device)
    else:
        device = "cpu"
        print("Using CPU.")

    # optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)

    # early stopping
    patience = args.patience
    old_validation_loss = float("Inf")

    model_name = f"{args.architecture}_l={args.layers}_es={args.embedding_size}_hs={args.hidden_size}_d={args.dropout}_e={args.epochs}_lr={args.learning_rate}_bs={args.batch_size}"

    if args.architecture == "transf":
        model_name += f"_ah={args.attention_heads}"

    # prepare for training
    epoch_training_loss = []
    epoch_validation_loss = []

    # training loop
    with trange(args.epochs, unit="epoch", ncols=100) as pbar:
        for e in pbar:
            pbar.set_description("Epoch %i" % (e + 1))

            mean_epoch_loss = 0
            batch_num = 0

            model.train()

            # make a pbar for each batch
            batch_pbar = tqdm(
                range(len(train_loader)),
                unit="batches",
                leave=False,
                desc="Batches",
                ncols=100,
            )

            for input, target in iter(train_loader):
                # get the labels
                target = target.to(device)

                # model prediction
                logits = model(input.to(device)).swapaxes(1, 2)

                # compute loss
                loss = loss_fn(input=logits, target=target)
                mean_epoch_loss += loss

                # propagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_num += 1

                # Update the progress bar for batches
                batch_pbar.update()

            # Close the progress bar for batches after each epoch
            batch_pbar.close()

            mean_epoch_loss /= batch_num
            mean_epoch_loss = float(mean_epoch_loss)
            # log
            epoch_training_loss.append(mean_epoch_loss)
            mean_epoch_loss = round(mean_epoch_loss, 6)
            mlflow.log_metric("mean_epoch_loss", mean_epoch_loss, step=e)

            # validation
            model.eval()
            with torch.no_grad():
                # get val data
                x_val, y_val = list(iter(val_loader))[0]
                y_val = y_val.to(device)
                # forward
                val_logits = model(x_val.to(device)).swapaxes(1, 2)
                # loss
                validation_loss = loss_fn(input=val_logits, target=y_val)
                validation_loss = float(validation_loss)
                # log
                epoch_validation_loss.append(validation_loss)
                validation_loss = round(validation_loss, 6)

                mlflow.log_metric("validation_loss", validation_loss, step=e)

                if validation_loss > old_validation_loss:
                    patience -= 1
                else:
                    patience = args.patience

                old_validation_loss = validation_loss

                if validation_loss <= min(epoch_validation_loss):
                    with tempfile.NamedTemporaryFile(
                        prefix=f"checkpoint_{e}", suffix=".pt"
                    ) as fp:
                        # use a temporary file to avoid cluttering the directory
                        torch.save(model.state_dict(), fp.name)
                        mlflow.log_artifact(fp.name, "state_dicts")

            pbar.set_postfix(train_loss=mean_epoch_loss, val_loss=validation_loss)

            if e % 5 == 0:
                with tempfile.NamedTemporaryFile(prefix=f"best", suffix=".pt") as fp:
                    # use a temporary file to avoid cluttering the directory
                    torch.save(model.state_dict(), fp.name)
                    mlflow.log_artifact(fp.name, "state_dicts")

            if patience == 0:
                mlflow.log_metric("early_stopping", True, step=e)
                break

    # TODO skapa en plot och spara som artificat i MLFlow
    plt.clf()
    plt.plot(epoch_training_loss, label="training loss")
    plt.plot(epoch_validation_loss, label="validation loss")
    plt.yscale("log")
    plt.legend()
    plt.xticks(range(0, args.epochs, max(1, args.epochs // 5)))
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    with tempfile.NamedTemporaryFile(suffix=".png") as temp:
        plt.savefig(temp.name, format="png")

        # Log the plot as an artifact in MLflow
        mlflow.log_artifact(temp.name, "plots")

    # finally, log the model
    mlflow.pytorch.log_model(model, "models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-hs", "--hidden_size", type=int, default=16)
    parser.add_argument("-es", "--embedding_size", type=int, default=16)
    parser.add_argument("-l", "--layers", type=int, default=2)
    parser.add_argument("-dp", "--dropout", type=float, default=0.2)
    parser.add_argument("-ah", "--attention_heads", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-bs", "--batch_size", type=int, default=100)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.9)
    parser.add_argument("-p", "--patience", type=int, default=5)
    parser.add_argument("-ld", "--load", type=str, default=None)
    parser.add_argument("-ml", "--max_sequence_length", type=int, default=256)
    parser.add_argument("-arch", "--architecture", type=str, default="lstm")
    parser.add_argument("-eo", "--epochs_offset", type=int, default=0)
    parser.add_argument("-n", "--sample_num", type=int, default=1)
    parser.add_argument("-ne", "--new_experiment", type=str, default=None)
    parser.add_argument("-en", "--experiment_name", type=str, default=None)
    parser.add_argument("-rs", "--random_seed", type=int, default=0)
    args = parser.parse_args()

    # TODO logga alla argument som parametrar i MLFlow
    assert args.architecture in [
        "lstm",
        "transf",
    ], "Architecture must be either 'lstm' or 'transf'"

    if args.architecture == "tranf":
        assert (
            args.attention_heads is not None
        ), "Attention heads must be specified for transformer architecture"

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

    # log all arguments as parameters
    with mlflow.start_run(experiment_id=experiment_id):
        for arg in vars(args):
            mlflow.log_param(arg, getattr(args, arg))

        main(args)

"""

python src/train.py --epochs 10

"""
