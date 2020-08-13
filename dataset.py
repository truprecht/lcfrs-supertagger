# pylint: disable=not-callable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, zeros, cat, unsqueeze
from torchtext.vocab import Vectors, FastText
from torch.nn.functional import softmax
from torch.utils.data import random_split, Subset

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
    def __init__(self, c, stoi, tag_distance=1):
        # TODO: sort by length
        self.dims = len(c.pos), len(c.preterms), len(c.supertags)
        self.sentence_embeddings = tuple( tensor([stoi(w) for w in sentence]) for sentence in c.sent_corpus )
        self.sentences = c.sent_corpus
        self.pos_tags = tuple( tensor(pos) for pos in c.pos_corpus )
        self.preterminals = tuple( tensor(prets) for prets in c.preterm_corpus )
        self.supertags = tuple( tensor(st) for st in c.supertag_corpus )
        self.supertag_distances = tensor(-tag_distance * c.confusion_matrix).softmax(dim=1).double()
        self.trees = c.tree_corpus

    def __getitem__(self, key):
        return self.sentences[key], self.sentence_embeddings[key], self.trees[key], \
            self.pos_tags[key], self.preterminals[key], self.supertags[key]

    def __len__(self):
        return len(self.sentences)

    def collate_training(self, results):
        (_, word_embeddings, _, pos, preterms, tags) = zip(*results)
        lens = tuple(len(sentence) for sentence in word_embeddings)
        tags = tuple(self.supertag_distances[tag] for tag in tags)
        return tuple(pad_sequence(batch, padding_value=-1) for batch in (word_embeddings, pos, preterms, tags)) + (tensor(lens),)

    def collate_test(self, results):
        (_, word_embeddings, _, pos, preterms, tags) = zip(*results)
        lens = tuple(len(sentence) for sentence in word_embeddings)
        return tuple(pad_sequence(batch, padding_value=-1) for batch in (word_embeddings, pos, preterms, tags)) + (tensor(lens),)

    def collate_val(self, results):
        (words, word_embeddings, trees, pos, preterms, tags) = zip(*results)
        lens = tuple(len(sentence) for sentence in words)
        return (words, trees) + tuple(pad_sequence(batch, padding_value=-1) for batch in (word_embeddings, pos, preterms, tags)) + (tensor(lens),)
