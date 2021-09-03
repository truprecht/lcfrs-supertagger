import torch

class BilstmEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers: int = 1, dropout: float = 0.0):
        super(BilstmEncoder, self).__init__()
        if layers > 0:
            self.bilstm = torch.nn.LSTM(input_dim, output_dim, bidirectional=True, num_layers=layers, dropout=dropout if layers > 1 else 0.0)
            self.combiner = torch.nn.Sequential(
                torch.nn.Linear(2*output_dim, output_dim, bias = False),
                torch.nn.ReLU(),
            )
            self.input_dim = input_dim
            self.output_dim = output_dim
        else:
            self.bilstm = None
            self.input_dim = input_dim
            self.output_dim = input_dim

    def forward(self, feats: torch.tensor):
        if not self.bilstm is None:
            feats, _ = self.bilstm(feats)
            return self.combiner(feats)
        return feats
