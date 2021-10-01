from enum import IntEnum

import torch as t
import flair
import numpy as np


class FfDecoder(t.nn.Module):
    def __init__(self, input_len: int, n_outputs: int, *args, **kwargs):
        super(FfDecoder, self).__init__()
        self.ff = t.nn.Linear(input_len, n_outputs)
        self._n_outputs = n_outputs

    def forward(self, feats, *args, **kwargs):
        return self.ff(feats)


class LstmDecoder(t.nn.Module):
    def __init__(self, input_len: int, n_outputs: int, embedding_len: int, lstm_size: int):
        super(LstmDecoder, self).__init__()
        self.output_embedding = t.nn.Embedding(n_outputs+1, embedding_len)
        self.lstm = t.nn.LSTMCell(input_len+embedding_len, lstm_size)
        self.ff = t.nn.Linear(lstm_size, n_outputs)
        self._n_outputs = n_outputs

    def single_forward(self, input: t.tensor, preds: t.tensor, state):
        inp = t.cat([input, self.output_embedding(preds)], dim=1)
        (h, c) = self.lstm(inp, state)
        scores = self.ff(h)
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


    def forward_beam(self, input: t.tensor, k: int = 1):
        batchlen = input.size(1)
        state = None
        preds = t.tensor([self._n_outputs]*batchlen, device=flair.device)

        scorelist = []
        idxlist = []
        for i, input_batch in enumerate(input):
            if i > 0:
                # repeat each input k times
                input_batch = input_batch.unsqueeze(1).expand(-1, k, -1).flatten(end_dim=1)
            scores, (hs, cs) = self.single_forward(input_batch, preds, state)
            # scores have shape (batchlen*k, output_len)
            if i > 0:
                scores = scores.reashape(batchlen, k*self._n_outputs)
            # scores have shape (batchlen, k*output_len)
            best_idc = t.from_numpy(np.argpartition(-scores.cpu(), k, axis=1)).to(flair.device)
            best_idc_predecessor = best_idc.div(k, rounding_mode="trunc").flatten() + t.arange(batchlen).unsqueeze(1).expand(-1, k).flatten()*k
            k_preds = best_idc % k
            preds = k_preds.flatten()
            state = hs[best_idc_predecessor], cs[best_idc_predecessor]
            idxlist.append(k_preds)
            scorelist.append(scores.gather(1, best_idc))
        return t.stack(scorelist), t.stack(idxlist)