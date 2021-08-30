from contextlib import AbstractContextManager
from functools import reduce
from os import makedirs
from pickle import dump, load
from tarfile import TarFile
from tempfile import TemporaryDirectory

from flair.datasets.sequence_labeling import ColumnCorpus, ColumnDataset

from discodop.treebank import READERS
from discodop.supertags.extraction import SupertagExtractor, GuideFactory, CompositionalNtConstructor
from discodop.supertags import SupertagGrammar

from tqdm import tqdm

from .parameters import Parameters
from .split import SplitFactory


corpusparam = Parameters(
    language=(str, None),
    filename=(str, None), split=(str, None), headrules=(str, ""), head_outward=(bool, False),
    guide=(str, "InorderGuide"), nonterminal_features=(str, "Constituent"), separate_attribs=(str, ""),
    inputenc=(str, "utf8"), inputfmt=(str, "export"), h=(int, 0), v=(int, 1), cachedir=(str, ".cache"))


def COLUMN_NAME_MAP(separate):
    return dict(enumerate(("text", "supertag") + separate))


class SupertagParseDataset(ColumnDataset):
    def __init__(self, path_prefix, separate_attribs = ()):
        super(SupertagParseDataset, self).__init__(f"{path_prefix}-tags", COLUMN_NAME_MAP(separate_attribs))
        self.separate_attribs = separate_attribs
        with open(f"{path_prefix}-trees") as file:
            for idx, tree in enumerate(file):
                tree = tree.strip()
                self[idx].add_label("tree", tree)


class SupertagParseCorpus(ColumnCorpus):
    def __init__(self, directory, separate_attribs = ()):
        super(SupertagParseCorpus, self).__init__(
            "", COLUMN_NAME_MAP(separate_attribs),
            train_file=f"{directory}/train-tags", test_file=f"{directory}/test-tags", dev_file=f"{directory}/dev-tags")
        self.separate_attribs = separate_attribs
        for s in ("train", "test", "dev"):
            with open(f"{directory}/{s}-trees") as file:
                for idx, tree in enumerate(file):
                    tree = tree.strip()
                    self.__getattribute__(s)[idx].add_label("tree", tree)

    @classmethod
    def extract_into(cls, dir: str, config: corpusparam, show_progress: bool = True):
        corpus = READERS[config.inputfmt](config.filename, encoding=config.inputenc, punct="move", headrules=config.headrules or None)
        extract = SupertagExtractor(
            GuideFactory(config.guide),
            CompositionalNtConstructor(config.nonterminal_features.split()),
            separate_attribs=config.separate_attribs.split(),
            headoutward=config.head_outward,
            horzmarkov=config.h, vertmarkov=config.v)

        split = SplitFactory().produce(config.split)

        tagfiles = { name: open(f"{dir}/{name}-tags", "w") for name in split.names() }
        treefiles = { name: open(f"{dir}/{name}-trees", "w") for name in split.names() }
        item_iterator = split.iter_items((item for _, item in corpus.itertrees()))
        if show_progress:
            item_iterator = tqdm(item_iterator, desc="Extracting supertag corpus", total=len(split))
        for name, item in item_iterator:
            supertags = list(extract(item.tree, keep=(name == "train")))
            for word, tag in zip(item.sent, supertags):
                tag, *atts = tag
                print(word, tag.str_tag(), *atts, file=tagfiles[name])
            print(file=tagfiles[name])
            print(item.tree, file=treefiles[name])
        dump(SupertagGrammar(tuple(extract.supertags), tuple(extract.roots)), open(f"{dir}/grammar", "wb"))

        for file in tagfiles.values():
            file.close()
        for file in treefiles.values():
            file.close()


class SupertagCorpusFile(AbstractContextManager):
    ARCF = ("train-tags", "dev-tags", "test-tags", "train-trees", "dev-trees", "test-trees", "grammar")

    def _archive_filename(self):
        basename = self.param.filename.split("/")[-1].split(".")[0]
        splitstr = "-".join(self.param.split.split())
        binstr = f"bin-{self.param.h}-{self.param.v}-{'ho' if self.param.head_outward else 'r'}"
        ntstr = "-".join(self.param.nonterminal_features.split())
        attribs = "supertags" + "".join(f"-{a}" for a in self.param.separate_attribs.split())
        return f"{self.param.cachedir}/{basename}.{splitstr}.{binstr}.{self.param.guide}.{ntstr}.{attribs}"


    def __init__(self, param: corpusparam):
        self._grammar = None
        self._corpus = None
        self.param = param
        self.tempdir = TemporaryDirectory()
        arc_filename = self._archive_filename()
        try:
            with TarFile(arc_filename) as arcfile:
                assert set(arcfile.getnames()) == set(self.__class__.ARCF), "damaged cache file"
                arcfile.extractall(self.tempdir.name)
        except FileNotFoundError:
            SupertagParseCorpus.extract_into(self.tempdir.name, self.param)
            makedirs(self.param.cachedir, exist_ok=True)
            with TarFile(arc_filename, "w") as tarf:
                for fname in self.__class__.ARCF:
                    tarf.add(f"{self.tempdir.name}/{fname}", arcname=fname)


    @property
    def grammar(self):
        if self._grammar is None:
            self._grammar = load(open(f"{self.tempdir.name}/grammar", "rb"))
        return self._grammar


    @property
    def corpus(self):
        if self._corpus is None:
            self._corpus = SupertagParseCorpus(self.tempdir.name, tuple(self.param.separate_attribs.split()))
        return self._corpus


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.tempdir.cleanup()