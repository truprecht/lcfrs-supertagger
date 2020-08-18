# pylint: disable=not-callable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, zeros, cat, unsqueeze, from_numpy, is_tensor, arange
from torchtext.vocab import Vectors, FastText
from torch.nn.functional import softmax
from torch.utils.data import random_split, Subset

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

def embedding_factory(def_str):
    def_strs = def_str.split()
    embedding_type, params = def_strs[0], def_strs[1:]
    if embedding_type == "word2vec":
        (filename,) = params
        return Vectors(filename)
    if embedding_type == "fasttext":
        (language,) = params
        return FastText(language)
    raise NotImplementedError()

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

    def evaluate(self, sequence, paramfilename=None):
        from discodop.eval import Evaluator, readparam
        from discodop.lexcorpus import to_parse_tree
        from discodop.lexgrammar import SupertagGrammar
        from discodop.tree import ParentedTree
        from discodop.treebanktransforms import removefanoutmarkers
        from discodop.treetransforms import unbinarize

        grammar = SupertagGrammar(self.corpus)
        evaluator = Evaluator(readparam(paramfilename))
        for i, (sent, gold, pos, preterms, supertags, weights) in enumerate(sequence):
            supertags = self.truncated_to_all.numpy()[supertags]
            cands = grammar.deintegerize_and_parse(sent, pos, preterms, supertags, weights, 1)
            try:
                cand = next(cands)
                cand = to_parse_tree(cand)
            except StopIteration:
                leaves = (f"({p} {i})" for p, i in zip(pos, range(len(sent))))
                cand = ParentedTree(f"(NOPARSE {' '.join(leaves)})")
            gold = unbinarize(removefanoutmarkers(gold))
            cand = unbinarize(removefanoutmarkers(cand))
            evaluator.add(i, gold, list(sent), cand, list(sent))
        evaluator.breakdowns()
        print(evaluator.summary())

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
        embedded_words = self.word_embeddings.get_vecs_by_tokens(words)
        pos = tensor(self.corpus.pos_corpus[key])
        preterms = tensor(self.corpus.preterm_corpus[key])
        supertags = tensor(self.corpus.supertag_corpus[key])
        tree = self.corpus.tree_corpus[key]
        return words, embedded_words, tree, \
            pos, preterms, self.all_to_truncated[supertags]

    def __len__(self):
        return len(self.corpus.sent_corpus)

    def collate_training(self, results):
        (_, word_embeddings, _, pos, preterms, tags) = zip(*results)
        lens = tuple(len(sentence) for sentence in word_embeddings)
        tag_probs = tuple( self.supertag_confusion[ts] for ts in tags )
        return tuple(pad_sequence(batch, padding_value=-2) for batch in (word_embeddings, pos, preterms, tag_probs)) + (tensor(lens),)

    def collate_test(self, results):
        (_, word_embeddings, _, pos, preterms, tags) = zip(*results)
        lens = tensor(tuple(len(sentence) for sentence in word_embeddings))
        word_embeddings, pos = pad_sequence(word_embeddings), pad_sequence(pos)
        preterms = pad_sequence(preterms, padding_value=-1)
        tags = pad_sequence(tags, padding_value=-2)
        return (word_embeddings, pos, preterms, tags, lens)

    def collate_val(self, results):
        (words, word_embeddings, trees, pos, _, _) = zip(*results)
        lens = tensor(tuple(len(sentence) for sentence in words))
        word_embeddings, pos = pad_sequence(word_embeddings), pad_sequence(pos)
        return (words, trees, word_embeddings, pos, tensor(lens))
