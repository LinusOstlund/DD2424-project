import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class MusicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_file: str,
        max_sequence_length: int,
        vocab_file: str,
    ):
        """
        Create the dataset. If `create_data==True`, `data_file`
        will be used to create the dataset, else the data and vocabs
        will be loaded from `data_file` and `vocab_file`.
        """
        super().__init__()

        self.max_sequence_length = max_sequence_length

        self._load_data(data_file, vocab_file)

    @property
    def sos_idx(self):
        return self.w2i["<sos>"]

    @property
    def eos_idx(self):
        return self.w2i["<eos>"]

    @property
    def pad_idx(self):
        return self.w2i["<pad>"]

    @property
    def vocab_size(self):
        return len(self.w2i)

    def _load_data(self, data_file: str, vocab_file: str):
        """
        Load data and vocab from json files.
        """
        with open(data_file, "r") as f:
            self.data = json.load(f)

        with open(vocab_file, "r") as f:
            data = json.load(f)
            self.w2i = data["w2i"]
            self.i2w = data["i2w"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.data[idx]["input"]),
            torch.LongTensor(self.data[idx]["target"]),
        )


"""
dataset = MusicDataset("data/dataset.txt", 128)
dataset = MusicDataset(
    data_file="data/dataset.json",
    vocab_file="data/vocab.json",
    max_sequence_length=128,
    create_data=False,
)
print(dataset, "\n", len(dataset))
print(dataset.w2i, "\n", dataset.i2w)
print(dataset[0])
"""
