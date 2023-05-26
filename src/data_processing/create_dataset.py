from tqdm import tqdm
import os
from collections import defaultdict
import json

ignore = [
    "^F,,",
    "e'",
    "^A,,",
    "=A",
    "=A,",
    "^G,,",
    "^C,",
    "^g",
    "=D",
    "=d",
    "^a",
    "g'",
    "B,,,,",
    "f'",
    "^d'",
    "c''",
    "a'",
    "^c'",
    "^D,,",
    "=A,,",
    "=G,",
    "=C",
    "=F",
    "g''",
    "b'",
    "=D,",
    "=a",
    "e''",
    "=F,",
    "^f'",
    "=G",
]

max_length = 256

data = ""
for path_to_file in tqdm(os.listdir("abc")):
    file_tokens = defaultdict(int)
    skip = False

    # open the file
    with open(f"abc/{path_to_file}", "r") as file:
        tokens = file.read().split(" ")

        # check for tokens to ignore
        for tok in ignore:
            if tok in tokens:
                skip = True
                break

        # skip the file
        if skip:
            continue

        # split file_tokens by space and remove empty strings
        tokens = list(filter(None, tokens))

        if len(tokens) > max_length:
            continue

        data += " ".join(tokens) + "\n"

with open("data/dataset.txt", "w") as f:
    f.write(data)


########### Moved this from create_dataset.py ############


def transform_data_to_json(data_file: str, max_sequence_length: int = 256) -> None:
    """
    Create dataset from `data_file`. In particular:
    - create w2i and i2w
    - create actual dataset input and target pairs
    """
    data = []
    w2i = defaultdict(int)
    i2w = defaultdict(str)

    print("Creating dataset...")
    with open("data/dataset.txt", "r") as f:
        # read the file and split in lines
        # 1 line = 1 piece
        lines = f.read().split("\n")

        # filter empty strings
        lines = list(filter(None, lines))

        # index pad, sos and eos tokens
        tok_id = 0
        w2i["<pad>"] = tok_id
        i2w[tok_id] = "<pad>"

        tok_id += 1
        w2i["<sos>"] = tok_id
        i2w[tok_id] = "<sos>"

        tok_id += 1
        w2i["<eos>"] = tok_id
        i2w[tok_id] = "<eos>"

        tok_id += 1

        print("Creating w2i and i2w...")
        # create w2i and i2w
        for piece in tqdm(lines):
            tokens = piece.split(" ")
            tokens = list(filter(None, tokens))

            for tok in tokens:
                # add token to dicts
                if not tok in w2i:
                    w2i[tok] = tok_id
                    i2w[tok_id] = tok
                    tok_id += 1

        # add the pieces to data using w2i
        # add <sos> at the start
        # add <eos> at the end
        # pad with <pad> until max length
        print("Indexing pieces...")
        for piece in tqdm(lines):
            tokens = piece.split(" ")
            tokens = list(filter(None, tokens))

            # input
            input = ["<sos>"] + tokens
            input = input[: max_sequence_length]

            # target
            target = tokens[: max_sequence_length - 1]
            target = target + ["<eos>"]

            # pad everything
            length = len(input)
            input.extend(["<pad>"] * (max_sequence_length - length))
            target.extend(["<pad>"] * (max_sequence_length - length))

            # convert to indexes
            input = [w2i.get(w) for w in input]
            target = [w2i.get(w) for w in target]

            # add dict to data
            item = {
                "input": input,
                "target": target,
            }
            data.append(item)

    # save data to json
    print("Writing data to json...")
    with open("data/dataset.json", "w") as f:
        data = json.dumps(data)
        f.write(data)

    # save vocabs to json
    print("Writing vocabs to json...")
    with open("data/vocab.json", "w") as f:
        vocab = {"w2i": w2i, "i2w": i2w}
        data = json.dumps(vocab)
        f.write(data)

    print("Done")

transform_data_to_json("data/dataset.txt")