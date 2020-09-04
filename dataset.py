# pylint: disable=not-callable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, zeros, cat, unsqueeze, from_numpy, is_tensor, arange
from torchtext.vocab import Vectors, FastText
from torch.nn.functional import softmax
from torch.utils.data import random_split, Subset
from collections import namedtuple

import numpy as np

def split_data(split_config, dataset):
    (random_or_serial, absolute_or_ratio, train, test, val) = split_config.split()

    if absolute_or_ratio == "ratio":
        train, test, val = (round(float(r) * len(dataset)) for r in (train, test, val))
    elif absolute_or_ratio == "absolute":
        train, test, val = int(train), int(test), int(val)
    else:
        raise NotImplementedError()

    if random_or_serial == "random":
        assert train + test + val <= len(dataset)
        return random_split(dataset, (train, test, val))
    elif random_or_serial == "serial":
        assert train + test + val <= len(dataset)
        starts = (0, train, train+test)
        ends = (train, train+test, train+test+val)
        return (Subset(dataset, range(start, end)) for start, end in zip(starts, ends))
    elif random_or_serial == "repeat":
        starts = (0, 0, 0)
        ends = (train, test, val)
        return (Subset(dataset, range(start, end)) for start, end in zip(starts, ends))
    else:
        raise NotImplementedError()

class TruncatedEmbedding(Vectors):
    """ Wrapper class around @Vectors that only considers a given vocabulary,
        all other vectors are dropped.
    """
    def __init__(self, embedding, vocab):
        self.itos = ()
        self.stoi = {}
        self.dim = embedding.dim
        self.unk_init = embedding.unk_init
        vectors = []
        for word in vocab:
            if word in embedding.stoi:
                self.stoi[word] = len(self.stoi)
                self.itos += (word,)
                vectors.append(embedding[word].unsqueeze(0))
        self.vectors = cat(vectors)

def embedding_factory(def_str, vector_cache):
    def_strs = def_str.split()
    embedding_type, params = def_strs[0], def_strs[1:]
    if embedding_type == "word2vec":
        (filename,) = params
        return Vectors(filename, cache=vector_cache)
    if embedding_type == "fasttext":
        (language,) = params
        return FastText(language, cache=vector_cache)
    raise NotImplementedError()

class characters(namedtuple("characters", "chars starts ends")):
    def to(self, *args):
        return characters(*(seq.to(*args) for seq in self))

class record(namedtuple("record", "inp out evl")):
    def to(self, *args):
        return record(
            tuple(seq.to(*args) for seq in self.inp),
            tuple(seq.to(*args) for seq in self.out),
            self.evl)
    def golds(self):
        sentlens = self.inp[-1]
        gprets, gtags = self.out
        for i, (sent, tree) in enumerate(zip(*self.evl)):
            yield gprets[:sentlens[i],i], gtags[:sentlens[i],i], sent, tree

class SupertagDataset(Dataset):
    def __init__(self, corpus, word_embeddings, tag_distance=1):
        # TODO: sort by length
        self.corpus = corpus
        self.word_embeddings = word_embeddings

        # store softmax-factor until initialized in @supertag_confusion
        self._supertag_confusion = tag_distance

        self.truncated_to_all = arange(len(self.corpus.supertags))
        self.all_to_truncated = arange(len(self.corpus.supertags))

    @property
    def dims(self):
        return (
            self.word_embeddings.dim,
            len(self.corpus.pos),
            len(self.corpus.preterms),
            len(self.truncated_to_all))

    @property
    def supertag_confusion(self):
        if not is_tensor(self._supertag_confusion):
            n_supertags = len(self.truncated_to_all)
            beta = self._supertag_confusion
            self._supertag_confusion = zeros((n_supertags, n_supertags)).double()
            self._supertag_confusion \
                = from_numpy(self.corpus.confusion_matrix[self.truncated_to_all][:, self.truncated_to_all])
            self._supertag_confusion = (-beta * self._supertag_confusion).softmax(dim=1)
        assert self._supertag_confusion.shape == (len(self.truncated_to_all), len(self.truncated_to_all))
        return self._supertag_confusion

    def evaluate(self, sequence, paramfilename=None, fallback=0.0, report=False):
        from discodop.eval import Evaluator, readparam
        from discodop.lexcorpus import to_parse_tree
        from discodop.lexgrammar import SupertagGrammar
        from discodop.tree import ParentedTree
        from discodop.treetransforms import unbinarize, removefanoutmarkers
        from re import match

        grammar = SupertagGrammar(self.corpus, fallback_prob=fallback)
        evaluator = Evaluator(readparam(paramfilename))
        ctags, cprets, npredictions = 0, 0, 0
        for i, (gprets, gtags, sent, gold, pos, preterms, supertags, weights) in enumerate(sequence):
            ctags += sum(1 for preds, gold in zip(supertags, gtags) if gold.item() in preds)
            cprets += sum(1 for pred, gold in zip(preterms, gprets) if gold.item() == pred)
            npredictions += len(sent)

            supertags = self.truncated_to_all.numpy()[supertags]
            cands = grammar.deintegerize_and_parse(sent, pos, preterms, supertags, weights, 1)
            try:
                cand = next(cands)
                cand = to_parse_tree(cand)
            except StopIteration:
                leaves = (f"({p} {i})" for p, i in zip(pos, range(len(sent))))
                cand = ParentedTree(f"(NOPARSE {' '.join(leaves)})")
            gold = unbinarize(removefanoutmarkers(gold.copy(deep=True)))
            cand = unbinarize(removefanoutmarkers(cand))
            evaluator.add(i, gold, list(sent), cand, list(sent))
        if report:
            evaluator.breakdowns()
            print(evaluator.summary())
        evlscores = { k: float_or_zero(v) for k,v in evaluator.acc.scores().items() }
        npredictions = max(1, npredictions)
        return {
            **evlscores,
            "acc/tags": ctags/npredictions,
            "acc/preterms": cprets/npredictions }

    def truncate_supertags(self, indices):
        """ Only consider those supertags occurring in sentences
            at given indices in corpus.
        """
        tags_in_trunc = set(t for index in indices for t in self.corpus.supertag_corpus[index])
        self.truncated_to_all = tensor(tuple(tags_in_trunc))
        self.all_to_truncated[:] = -1
        self.all_to_truncated[self.truncated_to_all] = arange(len(self.truncated_to_all))

    def __getitem__(self, key):
        words = self.corpus.sent_corpus[key]
        chars = tensor(list(" ".join(words).encode("utf8")))
        spaces = cat((tensor([-1]), (chars==32).nonzero(as_tuple=False).squeeze(-1), tensor([len(chars)])))
        embedded_words = self.word_embeddings.get_vecs_by_tokens(words, lower_case_backup=True)
        pos = tensor(self.corpus.pos_corpus[key])
        preterms = tensor(self.corpus.preterm_corpus[key])
        supertags = tensor(self.corpus.supertag_corpus[key])
        tree = self.corpus.tree_corpus[key]
        return record(
                (characters(chars, spaces[:-1]+1, spaces[1:]-1), embedded_words, pos),
                (preterms, self.all_to_truncated[supertags]),
                (words, tree))

    def __len__(self):
        return len(self.corpus.sent_corpus)

    def collate_common(self, results):
        results = tuple(zip(*results))
        clens = tensor([len(inp[0].chars) for inp in results[0]])
        slens = tensor([len(inp[1]) for inp in results[0]])
        cs, we, pos = zip(*results[0])
        chars = characters(*(pad_sequence(seq) for seq in zip(*cs)))
        we, pos = pad_sequence(we), pad_sequence(pos)
        preterms, tags = (pad_sequence(seq, padding_value=-1) for seq in zip(*results[1]))
        words, trees = zip(*results[2])
        return record(
            (chars, clens, we, pos, slens),
            (preterms, tags),
            (words, trees))

def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0
    except:
        return 0.0
