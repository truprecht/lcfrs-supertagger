from torch.nn import LSTM, Linear, Module, Embedding, Dropout, Sequential, ReLU
from torch import cat, zeros, ones, tensor, full
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from collections import namedtuple
from parameters import Parameters

class CharLstm(Module):
    hyperparam = Parameters(
        hidden_size=(int, 100), layers=(int, 1), dropout=(float, 0.1), embedding_dim=(int, 30))

    def __init__(self, param=None):
        if param is None:
            param = self.hyperparam.default()
        super(CharLstm, self).__init__()
        self.output_size = param.hidden_size * 4

        self.charembed = Embedding(256, param.embedding_dim)
        self.bilstm = LSTM(input_size=param.embedding_dim, hidden_size=param.hidden_size, \
            bidirectional=True, num_layers=param.layers, dropout=param.dropout if param.layers > 1 else 0)
    
    def forward(self, cs, clens):
        """ input shape: (characters + padding, batch,), (words + padding, batch,)
        """
        output_size = self.bilstm.hidden_size*2
        y, _ = self.bilstm(pack_padded_sequence(self.charembed(cs.chars), clens, enforce_sorted=False))
        y, _ = pad_packed_sequence(y)
        return cat((
            y.gather(0, cs.starts.unsqueeze(-1).expand(cs.starts.shape+(output_size,))),
            y.gather(0, cs.ends.unsqueeze(-1).expand(cs.ends.shape+(output_size,)))
        ), dim=2)

class tagger(Module):
    hyperparam = Parameters(
        pos_embedding_dim=(int, 10), dropout=(float, 0.1),
        char_lstm_size=(int, 100), char_lstm_layers=(int, 1), char_embedding_dim=(int, 30),
        lstm_layers=(int, 1), lstm_size=(int, 100),
        mlp_layers=(int, 1), mlp_size=(int, 100))

    def __init__(self, dims, param=None):
        super(tagger, self).__init__()
        if param is None:
            param = self.hyperparam.default()
        word_embedding_dim, postags, preterms, supertags = dims

        self.charbilstm = CharLstm(CharLstm.hyperparam(
            embedding_dim=param.char_embedding_dim, hidden_size=param.char_lstm_size,
            layers=param.char_lstm_layers, dropout=param.dropout))
        self.pos = Embedding(postags+1, param.pos_embedding_dim, padding_idx=0)

        word_embedding_size = self.charbilstm.output_size + word_embedding_dim + param.pos_embedding_dim
        self.bilstm = LSTM(input_size=word_embedding_size, hidden_size=param.lstm_size, \
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
    
    def forward(self, cs, clens, wordembeddings, postags, slens):
        """ input shape (words + padding, batch,), (words + padding, batch,), (batch,)
            output shape ((words + padding, batch, supertags), (words + padding, batch, preterms))
        """
        embedding = cat((
            self.charbilstm(cs, clens),
            wordembeddings,
            self.pos(postags + 1)), -1)
        x = pack_padded_sequence(embedding, slens, enforce_sorted=False)
        x, _ = self.bilstm(x)
        x, _ = pad_packed_sequence(x)
        x = self.ffs(x)
        return x.split((self.preterminals, self.supertags), dim=2)

    def train_loss(self, scores, gold):
        from torch.nn.functional import cross_entropy
        (cpt, cst, gpt, gst) = (t.flatten(end_dim=1) for t in (*scores, *gold))
        return cross_entropy(cpt, gpt, ignore_index=-1), cross_entropy(cst, gst, ignore_index=-1)

    def test_loss(self, scores, gold):
        from torch.nn.functional import nll_loss
        (cpt, cst, gpt, gst) = (t.flatten(end_dim=1) for t in (*scores, *gold))
        (cpt, cst) = (t.log_softmax(dim=1) for t in (cpt, cst))
        return nll_loss(cpt, gpt, ignore_index=-1), nll_loss(cst, gst, ignore_index=-1)

    def predict(self, cs, clens, wordembeddings, postags, slens, k=1):
        scores = self.forward(cs, clens, wordembeddings, postags, slens)
        pt, sts, ws = tagger.n_best_tags(scores, k)
        for idx, sentence_len in enumerate(slens):
            preterminals = pt[0:sentence_len, idx].cpu().numpy()
            supertags = sts[0:sentence_len, idx]
            weights = ws[0:sentence_len, idx]
            pos = postags[0:sentence_len, idx].cpu().numpy()
            yield pos, preterminals, supertags, weights

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
        padding = gold_indices == -2
        not_trained = gold_indices == -1
        # set padding in gold_indices to 0
        gold_indices = gold_indices * (~(padding | not_trained))
        gold_scores = y.gather(2, gold_indices.unsqueeze(2))
        gold_position = (y > gold_scores).sum(dim=2)
        gold_position[padding] = -2
        gold_position[not_trained] = -1
        return gold_position

def test_charlstm_dims():
    import torch

    sentences = [torch.tensor(list(s.encode("utf8"))) for s in ["I am a sentence .", "So am I ."]]
    x = torch.nn.utils.rnn.pad_sequence(sentences)
    spaces = [cat((tensor([-1]), torch.nonzero(s==32, as_tuple=False).squeeze(), tensor([len(s)]))) for s in sentences]
    p = CharLstm.hyperparam.default()
    starts = pad_sequence([ sp[:-1]+1 for sp in spaces ])
    ends = pad_sequence([ sp[1:]-1 for sp in spaces ])
    lens = tensor([len(s) for s in sentences])
    assert CharLstm(p).forward(x, starts, ends, lens).shape == (5, 2, p.hidden_size*4)


def test_dims():
    from torch import rand, randint, no_grad
    from torch.nn import Flatten

    words = 8
    word_dim = 10
    char_dim = 15
    pos_tags = 5
    supertags = 100
    preterms = 7
    batch_size = 20
    max_seq_size = 100
    max_word_len = 10

    word_embedding = rand((words, word_dim))
    t = tagger((word_dim, pos_tags, preterms, supertags),
            param=tagger.hyperparam(pos_embedding_dim=pos_tags, char_lstm_size=char_dim))
    words = word_embedding[randint(words, (max_seq_size, batch_size))]
    pos = randint(pos_tags, (max_seq_size, batch_size))
    lens = randint(max_seq_size, (batch_size,)) + 1
    lens[0] = max_seq_size
    wordlens = randint(max_word_len, (batch_size, max_seq_size)) + 1


    chars = [
        cat([ cat((randint(40, 100, (wordlen,)), tensor([32]))) for wordlen in sent[:slen] ])[:-1]
        for sent, slen in zip(wordlens, lens)
    ]
    spaces = [cat((tensor([-1]), (cs==32).nonzero(as_tuple=False).squeeze(), tensor([len(cs)]))) for cs in chars]
    starts = pad_sequence([ sps[:-1]+1 for sps in spaces ])
    ends = pad_sequence([ sps[1:]-1 for sps in spaces ])
    charlens = tensor([ len(cs) for cs in chars ])
    chars = pad_sequence(chars)

    for batch in range(1, batch_size):
        for oob in range(lens[batch], max_seq_size):
            words[oob, batch] = -1
            pos[oob, batch] = -1

    (pt, st) = t((chars, starts, ends, charlens, words, pos, lens))
    mask = t.get_mask(lens)
    assert st.shape == (max_seq_size, batch_size, supertags) and pt.shape == (max_seq_size, batch_size, preterms)
    assert st[mask].shape == (lens.sum(), supertags) and pt[mask].shape == (lens.sum(), preterms)

    with no_grad():
        gold_tags = randint(supertags, (max_seq_size, batch_size))
        gold_tags[pos == -1] = -1
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