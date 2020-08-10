from torch.nn import LSTM, Linear, Module, Embedding
from torch import cat, zeros, ones
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class tagger(Module):
    def __init__(self, dims, pretrained_word_embedding, pos_embedding=10, lstm_layers=1, dropout=0, hidden_size=100):
        super(tagger, self).__init__()
        poss, preterms, supertags = dims
        word_embedding_dim = pretrained_word_embedding.shape[1]
        # add padding entry to word embedding at index 0
        padding_entry = zeros((1, pretrained_word_embedding.shape[1]))
        pretrained_word_embedding = cat((padding_entry, pretrained_word_embedding))
        self.words = Embedding.from_pretrained(pretrained_word_embedding, padding_idx=0)
        self.pos = Embedding(poss+1, pos_embedding, padding_idx=0)
        self.bilstm = LSTM(input_size=word_embedding_dim+pos_embedding, hidden_size=hidden_size, bidirectional=True, num_layers=lstm_layers, dropout=dropout)
        self.ff_st = Linear(in_features=2*hidden_size, out_features=supertags)
        self.ff_pt = Linear(in_features=2*hidden_size, out_features=preterms)

    def get_mask(self, lens):
        m = tuple(ones((seqlen,), dtype=bool) for seqlen in lens)
        return pad_sequence(m, padding_value=False)
    
    def forward(self, x):
        """ input shape (words + padding, batch,), (words + padding, batch,), (batch,)
            output shape ((words + padding, batch, supertags), (words + padding, batch, preterms))
        """
        (words, pos_tags, lens) = x
        pos_tags = self.pos(pos_tags+1)
        words = self.words(words+1)
        x = cat((words, pos_tags), 2)
        x = pack_padded_sequence(x, lens, enforce_sorted=False)
        x, _ = self.bilstm(x)
        x, lens = pad_packed_sequence(x)
        st_scores = self.ff_st(x)
        pt_scores = self.ff_pt(x)
        return (pt_scores, st_scores)

    @classmethod
    def n_best_tags(cls, y, n):
        from numpy import argpartition
        pt_scores, st_scores = y
        pts = pt_scores.argmax(dim=1)
        sts = argpartition(-st_scores, n, axis=1)[:,0:n]
        return (pts, sts)

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
    from torch import rand, randint
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
    assert st.shape == (max_seq_size, batch_size, supertags) and pt.shape == (max_seq_size, batch_size, preterms)
    iron = Flatten(0, 1)
    assert iron(st).shape == (max_seq_size * batch_size, supertags) and iron(pt).shape == (max_seq_size * batch_size, preterms)

    gold_tags = randint(supertags, (max_seq_size, batch_size))
    gold_tags[words == -1] = -1
    assert tagger.index_in_sorted(st, gold_tags).shape == (max_seq_size, batch_size)
    (best_pt, n_best_st) = tagger.n_best_tags((pt, st), 3)
    assert best_pt.shape == (max_seq_size, batch_size) and n_best_st.shape == (max_seq_size, batch_size, 3)

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