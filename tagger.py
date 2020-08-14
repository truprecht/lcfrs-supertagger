from torch.nn import LSTM, Linear, Module, Embedding, Dropout, Sequential, ReLU
from torch import cat, zeros, ones
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from collections import namedtuple

class tagger(Module):
    class Hyperparameters(namedtuple("Hyperparameters", "pos lstm_layers lstm_size mlp_layers mlp_size dropout")):
        defaults = (10, 1, 100, 1, 100, 0.1)
        types = (int, int, int, int, int, float)

        @classmethod
        def default(cls):
            return cls(*cls.defaults)

        @classmethod
        def from_dict(cls, dic):
            values = (val_type(dic[key]) if key in dic else default \
                for (default, val_type, key) in zip(cls.defaults, cls.types, cls._fields))
            return cls(*values)

    def __init__(self, dims, param=None):
        super(tagger, self).__init__()
        if param is None:
            param = self.Hyperparameters.default()
        word_embedding_dim, postags, preterms, supertags = dims

        self.pos = Embedding(postags+1, param.pos, padding_idx=0)

        self.bilstm = LSTM(input_size=word_embedding_dim+param.pos, hidden_size=param.lstm_size, \
            bidirectional=True, num_layers=param.lstm_layers, dropout=param.dropout if param.lstm_layers > 1 else 0)
        
        next_size = 2*param.lstm_size
        ffs = []
        for _ in range(param.mlp_layers-1):
            ffs.append(Dropout(p=param.dropout))
            ffs.append(Linear(in_features=next_size, out_features=param.mlp_size))
            ffs.append(ReLU())
            next_size = param.mlp_size

        ffs.append(Dropout(p=param.dropout))
        ffs.append(Linear(in_features=next_size, out_features=preterms+supertags))
        self.ffs = Sequential(*ffs)
        self.preterminals = preterms
        self.supertags = supertags

    def get_mask(self, lens):
        m = tuple(ones((seqlen,), dtype=bool) for seqlen in lens)
        return pad_sequence(m, padding_value=False)
    
    def forward(self, x):
        """ input shape (words + padding, batch,), (words + padding, batch,), (batch,)
            output shape ((words + padding, batch, supertags), (words + padding, batch, preterms))
        """
        (words, pos_tags, lens) = x
        pos_tags = self.pos(pos_tags+1)
        x = cat((words, pos_tags), 2)
        x = pack_padded_sequence(x, lens, enforce_sorted=False)
        x, _ = self.bilstm(x)
        x, lens = pad_packed_sequence(x)
        x = self.ffs(x)
        return x.split((self.preterminals, self.supertags), dim=2)

    @classmethod
    def n_best_tags(cls, y, n):
        import numpy as np
        pt_scores, st_scores = y
        pts = pt_scores.argmax(dim=2)
        sts = np.argpartition(-st_scores.cpu().numpy(), n-1, axis=2)[:,:,0:n]
        weights = np.take_along_axis(st_scores.cpu().numpy(), sts, 2)
        # softmax for weights
        # TODO move this to model layer
        weights = np.exp(weights)
        weights = weights / weights.sum(axis=2)[:,:,np.newaxis]
        weights = -np.log(weights)
        return (pts, sts, weights)

    @classmethod
    def index_in_sorted(cls, y, gold_indices):
        padding = gold_indices == -1
        # set padding in gold_indices to 0
        gold_indices = gold_indices * (~padding)
        gold_scores = y.gather(2, gold_indices.unsqueeze(2))
        gold_position = (y > gold_scores).sum(dim=2)
        gold_position[padding] = -1
        return gold_position

def test_dims():
    from torch import rand, randint, no_grad
    from torch.nn import Flatten

    words = 8
    word_dim = 10
    pos_tags = 5
    supertags = 100
    preterms = 7
    batch_size = 20
    max_seq_size = 100

    word_embedding = rand((8, word_dim))
    t = tagger((pos_tags, preterms, supertags), word_embedding)
    words = randint(words, (max_seq_size, batch_size))
    pos = randint(pos_tags, (max_seq_size, batch_size))
    lens = randint(max_seq_size, (batch_size,)) + 1
    lens[0] = max_seq_size
    for batch in range(1, batch_size):
        for oob in range(lens[batch], max_seq_size):
            words[oob, batch] = -1
            pos[oob, batch] = -1

    (pt, st) = t((words, pos, lens))
    mask = t.get_mask(lens)
    assert st.shape == (max_seq_size, batch_size, supertags) and pt.shape == (max_seq_size, batch_size, preterms)
    assert st[mask].shape == (lens.sum(), supertags) and pt[mask].shape == (lens.sum(), preterms)

    with no_grad():
        gold_tags = randint(supertags, (max_seq_size, batch_size))
        gold_tags[words == -1] = -1
        assert tagger.index_in_sorted(st, gold_tags).shape == (max_seq_size, batch_size)
        (best_pt, n_best_st, n_best_weights) = tagger.n_best_tags((pt, st), 3)
        assert best_pt.shape == (max_seq_size, batch_size) \
            and n_best_st.shape == (max_seq_size, batch_size, 3) \
            and n_best_weights.shape == (max_seq_size, batch_size, 3)

def test_index_in_sorted():
    from torch import rand, randint

    supertags = 100
    batch_size = 20
    max_seq_size = 100

    gold = randint(supertags, (max_seq_size, batch_size))
    scores = rand((max_seq_size, batch_size, supertags))
    lens = randint(max_seq_size, (batch_size,)) + 1
    lens[0] = max_seq_size
    for batch in range(1, batch_size):
        for oob in range(lens[batch], max_seq_size):
            gold[oob, batch] = -1

    positions = tagger.index_in_sorted(scores, gold)
    for batch in range(1, batch_size):
        for word in range(0, lens[batch]):
            p = positions[word, batch]
            sorted_scores, sorted_indices = scores[word, batch].sort(descending=True)
            assert scores[word, batch, gold[word, batch]] == sorted_scores[p]
            assert sorted_indices[p] == gold[word, batch]
    
    for batch in range(1, batch_size):
        for oob in range(lens[batch], max_seq_size):
            assert positions[oob, batch] == -1