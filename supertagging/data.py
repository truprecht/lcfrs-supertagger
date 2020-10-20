from flair.datasets.sequence_labeling import ColumnCorpus
from .parameters import Parameters

corpusparam = Parameters(
    language=(str, None),
    filename=(str, None), split=(str, None),
    inputenc=(str, "utf8"), inputfmt=(str, "export"), h=(int, 0), v=(int, 1))

class SupertagParseCorpus(ColumnCorpus):
    def __init__(self, basename):
        super(SupertagParseCorpus, self).__init__(
            "", {0: "text", 1: "pos", 2: "supertag"},
            train_file=f"{basename}.train.tags", test_file=f"{basename}.test.tags", dev_file=f"{basename}.dev.tags")
        for s in  ("train", "test", "dev"):
            with open(f"{basename}.{s}.trees") as file:
                for idx, tree in enumerate(file):
                    tree = tree.strip()
                    self.__getattribute__(s)[idx].add_label("tree", tree)