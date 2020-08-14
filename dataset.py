# pylint: disable=not-callable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, zeros, cat, unsqueeze, from_numpy
from torchtext.vocab import Vectors, FastText
from torch.nn.functional import softmax
from torch.utils.data import random_split, Subset

import numpy as np

def split_data(split_config, dataset):
    (random_or_serial, absolute_or_ratio, train, test, val) = split_config.split()
    if absolute_or_ratio == "ratio":
        train, test, val = float(train), float(test), float(val)
        assert train + test + val <= 1
        train, test, val = round(train * len(dataset)), round(test * len(dataset)), round(val * len(dataset))
    elif absolute_or_ratio == "absolute":
        train, test, val = int(train), int(test), int(val)
        assert train + test + val <= len(dataset)
    if random_or_serial == "random":
        return random_split(dataset, (train, test, val))
    elif random_or_serial == "serial":
        starts = (0, train, train+test)
        ends = (train, train+test, train+test+val)
        return (Subset(dataset, range(start, end)) for start, end in zip(starts, ends))

class TruncatedEmbedding(Vectors):
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
    def __init__(self, c, word_embeddings, tag_distance=1):
        # TODO: sort by length
        self.dims = word_embeddings.dim, len(c.pos), len(c.preterms), len(c.supertags)

        self.trees = c.tree_corpus
        self.sentences = c.sent_corpus
        self.word_embeddings = word_embeddings

        self.pos_tags = tuple( tensor(pos) for pos in c.pos_corpus )
        self.preterminals = tuple( tensor(prets) for prets in c.preterm_corpus )
        self.supertags = tuple( np.array(st) for st in c.supertag_corpus )
        self.supertag_distances = tensor(-tag_distance * c.confusion_matrix).softmax(dim=1).double()

        self.truncated_to_all = np.arange(len(self.supertags))
        self.all_to_truncated = np.arange(len(self.supertags))
        self.truncated_distances = self.supertag_distances

    def truncate_supertags(self, indices):
        """ Only consider those supertags occurring in sentences
            at given indices in corpus.
        """
        tags_in_trunc = set(t for index in indices for t in self.supertags[index])
        self.truncated_to_all = np.fromiter(iter(tags_in_trunc), dtype=int, count=len(tags_in_trunc))
        self.all_to_truncated = np.full((self.dims[3],), len(self.truncated_to_all))
        self.all_to_truncated[self.truncated_to_all] = np.arange(len(self.truncated_to_all))

        # add one unk-supertag
        n_supertags = len(self.truncated_to_all)+1
        self.dims = (*self.dims[0:3], n_supertags)

        self.truncated_distances = zeros((n_supertags, n_supertags)).double()
        self.truncated_distances[0:n_supertags-1, 0:n_supertags-1] \
            = self.supertag_distances[self.truncated_to_all][:, self.truncated_to_all]
        self.truncated_distances /= self.truncated_distances.sum(dim=1).unsqueeze(1)

    def __getitem__(self, key):
        words = self.sentences[key]
        truncated_tags = from_numpy(self.all_to_truncated[self.supertags[key]])
        return words, self.word_embeddings.get_vecs_by_tokens(words), self.trees[key], \
            self.pos_tags[key], self.preterminals[key], truncated_tags

    def __len__(self):
        return len(self.sentences)

    def collate_training(self, results):
        (_, word_embeddings, _, pos, preterms, tags) = zip(*results)
        lens = tuple(len(sentence) for sentence in word_embeddings)
        tag_probs = tuple( self.truncated_distances[ts] for ts in tags )
        return tuple(pad_sequence(batch, padding_value=-1) for batch in (word_embeddings, pos, preterms, tag_probs)) + (tensor(lens),)

    def collate_test(self, results):
        (_, word_embeddings, _, pos, preterms, tags) = zip(*results)
        lens = tuple(len(sentence) for sentence in word_embeddings)
        return tuple(pad_sequence(batch, padding_value=-1) for batch in (word_embeddings, pos, preterms, tags)) + (tensor(lens),)

    def collate_val(self, results):
        (words, word_embeddings, trees, pos, _, _) = zip(*results)
        lens = tuple(len(sentence) for sentence in words)
        return (words, trees) + tuple(pad_sequence(batch, padding_value=-1) for batch in (word_embeddings, pos)) + (tensor(lens),)
