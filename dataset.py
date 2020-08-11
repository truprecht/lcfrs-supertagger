# pylint: disable=not-callable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, zeros
from torchtext.vocab import Vectors, FastText
from torch.nn.functional import softmax

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
        self.sentences = tuple( tensor([stoi(w) for w in sentence]) for sentence in c.sent_corpus )
        self.pos_tags = tuple( tensor(pos) for pos in c.pos_corpus )
        self.preterminals = tuple( tensor(prets) for prets in c.preterm_corpus )
        self.supertags = tuple( tensor(st) for st in c.supertag_corpus )
        self.supertag_distances = tensor(-tag_distance * c.confusion_matrix).softmax(dim=1).double()

    def __getitem__(self, key):
        return self.sentences[key], self.pos_tags[key], \
            self.preterminals[key], self.supertags[key]

    def __len__(self):
        return len(self.sentences)

    def collate_training(self, results):
        (words, pos, preterms, tags) = zip(*results)
        lens = tuple(len(sentence) for sentence in words)
        tags = tuple(self.supertag_distances[tag] for tag in tags)
        return tuple(pad_sequence(batch, padding_value=-1) for batch in (words, pos, preterms, tags)) + (tensor(lens),)

    def collate_test(self, results):
        (words, pos, preterms, tags) = zip(*results)
        lens = tuple(len(sentence) for sentence in words)
        return tuple(pad_sequence(batch, padding_value=-1) for batch in (words, pos, preterms, tags)) + (tensor(lens),)