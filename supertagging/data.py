from flair.datasets.sequence_labeling import ColumnCorpus, ColumnDataset
from .parameters import Parameters

corpusparam = Parameters(
    language=(str, None),
    filename=(str, None), split=(str, None), headrules=(str, ""),
    propterm_nonterminals=(str, "strict"), propterm_marker=(str, ""), split_strictness=(str, "strict"),
    inputenc=(str, "utf8"), inputfmt=(str, "export"), h=(int, 0), v=(int, 1))

COLUMN_NAME_MAP = {0: "text", 1: "pos", 2: "supertag"}

class SupertagParseDataset(ColumnDataset):
    def __init__(self, path):
        super(SupertagParseDataset, self).__init__(f"{path}.tags", COLUMN_NAME_MAP)
        with open(f"{path}.trees") as file:
            for idx, tree in enumerate(file):
                tree = tree.strip()
                self[idx].add_label("tree", tree)

class SupertagParseCorpus(ColumnCorpus):
    def __init__(self, basename):
        super(SupertagParseCorpus, self).__init__(
            "", COLUMN_NAME_MAP,
            train_file=f"{basename}.train.tags", test_file=f"{basename}.test.tags", dev_file=f"{basename}.dev.tags")
        for s in  ("train", "test", "dev"):
            with open(f"{basename}.{s}.trees") as file:
                for idx, tree in enumerate(file):
                    tree = tree.strip()
                    self.__getattribute__(s)[idx].add_label("tree", tree)