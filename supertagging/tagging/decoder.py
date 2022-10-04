from enum import IntEnum

import torch as t
import flair
import numpy as np


# a linear layer for a multiclass prediction
class FfDecoder(t.nn.Module):
    def __init__(self, input_len: int, n_outputs: int, *args, **kwargs):
        super(FfDecoder, self).__init__()
        self.ff = t.nn.Linear(input_len, n_outputs)
        self._n_outputs = n_outputs

    def forward(self, feats, *args, **kwargs):
        return self.ff(feats)


# Combines the input vectors with the output of an LSTM language model
# the outputs are predicted from left to right, previous positions are embedded and fed into the lm
class LmDecoder(t.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, language_model_embedding_dim: int, language_model_hidden_dim: int):
        super(LmDecoder, self).__init__()
        assert language_model_hidden_dim > 0 and language_model_embedding_dim > 0
        
        self.language_model_embedding = t.nn.Embedding(output_dim+1, language_model_embedding_dim)
        self.language_model = t.nn.LSTMCell(language_model_embedding_dim, language_model_hidden_dim)

        combiner_dim = max(input_dim, language_model_hidden_dim)
        self.combiner = t.nn.Sequential(
            t.nn.Linear(input_dim+language_model_hidden_dim, combiner_dim, bias=False),
            t.nn.ReLU()
        )

        self.ff = t.nn.Linear(combiner_dim, output_dim)
        self._n_outputs = output_dim


    # a step for a single position in a batch:
    # * embeds the previous prediction `preds`,
    # * puts the embedding into the language model,
    # * combines the input with the language model vector and
    # * computes prediction scores via a linear layer
    def single_forward(self, input: t.tensor, preds: t.tensor, state):
        h, c = self.language_model(self.language_model_embedding(preds), state)
        scores = self.ff(self.combiner(t.cat([input, h], dim=1)))
        return scores, (h, c)


    def forward(self, input: t.tensor, gold_outputs: t.tensor = None, gold_sampling_prob: float = 0.0):
        batchlen = input.size(1)
        state = None
        preds = t.tensor([self._n_outputs]*batchlen, device=flair.device)

        scorelist = []
        for i, input_batch in enumerate(input):
            scores, state = self.single_forward(input_batch, preds, state)
            if gold_sampling_prob == 0.0:
                preds = scores.argmax(dim=1)
            elif gold_sampling_prob == 1.0:
                preds = gold_outputs[i]
            else:
                preds = t.where((t.rand(batchlen) < gold_sampling_prob).to(flair.device), gold_outputs[i], scores.argmax(dim=1))
            scorelist.append(scores)
        return t.stack(scorelist)


# Decoder based on the transformer decoder model
class TfDecoder(t.nn.Module):
    def __init__(self, input_len: int, n_outputs: int, *args, **kwargs):
        pass

    def forward(self, feats, *args, **kwargs):
        pass