import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        embedding_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm_layer = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.multiplier = 2 if bidirectional else 1
        self.prediction_layer = nn.Linear(self.multiplier * hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    # batch (B, L) -> (B * L, V)
    def forward(self, batch):
        B = batch.size(0)
        L = batch.size(1)
        H = self.hidden_size
        V = self.vocab_size

        # (B, L) -> (B, L, E)
        embedding = self.embedding(batch)

        # (B, L, E) -> (B, L, H)
        lstm_out, _ = self.lstm_layer(embedding)

        # (B, L, H) -> (B * L, H)
        pred_input = lstm_out.reshape(B * L, self.multiplier * H)

        # (B * L, H) -> (B * L, V)
        logits = self.prediction_layer(pred_input)

        # (B * L, V) -> (B, L, V)
        # logits = logits.reshape(B, L, V)

        return logits

    def inference(self, sos_idx, eos_idx, max_len=100, mode="greedy", device="cpu"):
        assert mode in ["greedy", "topp", "topk"]

        with torch.no_grad():
            generation = [sos_idx]
            t = 0
            while t < max_len:
                input = torch.LongTensor([generation[-1]]).to(device).unsqueeze(0)

                out = self.forward(input)

                out = self.softmax(out)

                tok = torch.argmax(out)
                generation.append(tok.item())

                if tok == eos_idx:
                    break

                t += 1

            return generation
