from flair.datasets.sequence_labeling import ColumnCorpus, ColumnDataset
from .parameters import Parameters

corpusparam = Parameters(
    language=(str, None),
    filename=(str, None), split=(str, None), headrules=(str, ""),
    guide=(str, "InorderGuide"), nonterminal_features=(str, "Constituent"), separate_attribs=(str, ""),
    inputenc=(str, "utf8"), inputfmt=(str, "export"), h=(int, 0), v=(int, 1))

def COLUMN_NAME_MAP(separate):
    return dict(enumerate(("text", "supertag") + separate))

class SupertagParseDataset(ColumnDataset):
    def __init__(self, path, separate_attribs = ()):
        super(SupertagParseDataset, self).__init__(f"{path}.tags", COLUMN_NAME_MAP(separate_attribs))
        self.separate_attribs = separate_attribs
        with open(f"{path}.trees") as file:
            for idx, tree in enumerate(file):
                tree = tree.strip()
                self[idx].add_label("tree", tree)

class SupertagParseCorpus(ColumnCorpus):
    def __init__(self, basename, separate_attribs = ()):
        super(SupertagParseCorpus, self).__init__(
            "", COLUMN_NAME_MAP(separate_attribs),
            train_file=f"{basename}.train.tags", test_file=f"{basename}.test.tags", dev_file=f"{basename}.dev.tags")
        self.separate_attribs = separate_attribs
        for s in  ("train", "test", "dev"):
            with open(f"{basename}.{s}.trees") as file:
                for idx, tree in enumerate(file):
                    tree = tree.strip()
                    self.__getattribute__(s)[idx].add_label("tree", tree)