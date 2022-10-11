import torch
import math


class NoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats):
        return feats


class BilstmEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers: int = 1, dropout: float = 0.0):
        assert layers >= 0
        super(BilstmEncoder, self).__init__()
        self.bilstm = torch.nn.LSTM(input_dim, output_dim, bidirectional=True, num_layers=layers, dropout=dropout if layers > 1 else 0.0)
        self.combiner = torch.nn.Sequential(
            torch.nn.Linear(2*output_dim, output_dim, bias = False),
            torch.nn.ReLU(),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, feats: torch.tensor):
        feats, _ = self.bilstm(feats)
        return self.combiner(feats)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers: int = 1, dropout: float = 0.0):
        assert layers >= 0 and input_dim == output_dim
        super().__init__()
        self.pos = PositionalEncoding(input_dim, dropout)
        self.encode = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(input_dim, nhead=4, dropout=dropout),
            layers,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, feats: torch.tensor):
        return self.encode(self.pos(feats))