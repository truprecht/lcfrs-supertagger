from contextlib import AbstractContextManager
from functools import reduce
from os import makedirs
from pickle import dump, load
import tarfile
from tempfile import TemporaryDirectory
from typing import Set
from enum import IntEnum
from functools import reduce
from operator import or_

from flair.datasets.sequence_labeling import ColumnCorpus, ColumnDataset
from flair.datasets.base import find_train_dev_test_files
from flair.data import Corpus

from discodop.treebank import READERS
from discodop.supertags.extraction import SupertagExtractor, GuideFactory, CompositionalNtConstructor
from discodop.supertags import SupertagGrammar, Supertag, LexicalRule

from tqdm import tqdm

from .model import parse_or_none
from .parameters import Parameters
from .split import SplitFactory


corpusparam = Parameters(
    language=(str, None),
    filename=(str, None), split=(str, None), headrules=(str, ""), head_outward=(bool, False),
    guide=(str, "InorderGuide"), nonterminal_features=(str, "Constituent"), core_attribs=(str, "grammar_rules transport constituent pos"),
    inputenc=(str, "utf8"), inputfmt=(str, "export"), h=(int, 0), v=(int, 1), cachedir=(str, ".cache"))


class supertag_attribs(IntEnum):
    grammar_rule = 2**0
    constituent = 2**1
    transport = 2**2
    pos = 2**3

    @classmethod
    def from_str(cls, s: str):
        short_to_long = { a.name[:1]: a for a in cls }
        return reduce(or_, (short_to_long[c] for c in s), 0)


COLUMN_NAME_MAP = dict(enumerate(("text", "grammar_rule", "constituent", "transport", "pos", "root_nonterminal")))


def token_to_supertag(tok, core: int):
    constituent = parse_or_none(tok.get_tag("constituent").value) if core & supertag_attribs.constituent else None
    transport = parse_or_none(tok.get_tag("transport").value, int) if core & supertag_attribs.transport else None
    pos = parse_or_none(tok.get_tag("pos").value) if core & supertag_attribs.pos else None
    return Supertag(
        LexicalRule(tok.get_tag("grammar_rule").value),
        constituent = constituent,
        transport = transport,
        pos = pos)


class SupertagParseDataset(ColumnDataset):
    def _init_supertags(self, core_attribs: int):
        for sentence in self:
            for token in sentence:
                tag = token_to_supertag(token, core_attribs)
                token.add_tag("supertag", tag.str_tag())

    def __init__(self, path, core_attribs: int):
        assert str(path).endswith("-tags")
        super(SupertagParseDataset, self).__init__(path, COLUMN_NAME_MAP)
        with open(f"{str(path)[:-5]}-trees") as file:
            for idx, tree in enumerate(file):
                tree = tree.strip()
                self[idx].add_label("tree", tree)
        self._init_supertags(core_attribs)
        self._core_attribs = core_attribs


class SupertagParseCorpus(ColumnCorpus):
    def __init__(self, directory, core_attribs: int):
        dev, test, train = (f"{directory}/{p}-tags" for p in ("dev", "test", "train"))
        train = SupertagParseDataset(train, core_attribs)
        dev = SupertagParseDataset(dev, core_attribs)
        test = SupertagParseDataset(test, core_attribs)
        Corpus.__init__(self, train, dev, test, name=directory)
        self._core_attribs = core_attribs

    @property
    def core_attribs(self):
        for attr in supertag_attribs:
            if self._core_attribs & attr:
                yield attr.name

    @property
    def separate_attribs(self):
        for attr in supertag_attribs:
            if not self._core_attribs & attr:
                yield attr.name

    @classmethod
    def extract_into(cls, dir: str, config: corpusparam, show_progress: bool = True):
        corpus = READERS[config.inputfmt](config.filename, encoding=config.inputenc, punct="move", headrules=config.headrules or None)
        extract = SupertagExtractor(
            GuideFactory(config.guide),
            CompositionalNtConstructor(config.nonterminal_features.split()),
            headoutward=config.head_outward,
            horzmarkov=config.h, vertmarkov=config.v)

        split = SplitFactory().produce(config.split)

        tagfiles = { name: open(f"{dir}/{name}-tags", "w") for name in split.names() }
        treefiles = { name: open(f"{dir}/{name}-trees", "w") for name in split.names() }
        item_iterator = split.iter_items((item for _, item in corpus.itertrees()))
        if show_progress:
            item_iterator = tqdm(item_iterator, desc="Extracting supertag corpus", total=len(split))
        for name, item in item_iterator:
            supertags = extract(item.tree)
            for word, (tag, is_root_node) in zip(item.sent, supertags):
                print(word, *(c if type(c) is str else repr(c) for c in tag), tag.rule.lhs if is_root_node else "None", file=tagfiles[name])
            print(file=tagfiles[name])
            print(item.tree, file=treefiles[name])

        for file in tagfiles.values():
            file.close()
        for file in treefiles.values():
            file.close()


class SupertagCorpusFile(AbstractContextManager):
    ARCF = ("train-tags", "dev-tags", "test-tags", "train-trees", "dev-trees", "test-trees")

    def _archive_filename(self):
        basename = self.param.filename.split("/")[-1].split(".")[0]
        splitstr = "-".join(self.param.split.split())
        binstr = f"bin-{self.param.h}-{self.param.v}-{'ho' if self.param.head_outward else 'r'}"
        ntstr = "-".join(self.param.nonterminal_features.split())
        return f"{self.param.cachedir}/{basename}.{splitstr}.{binstr}.{self.param.guide}.{ntstr}.tar.gz"


    def check_parameters(self):
        if self.param.guide == "ModifierGuide" and \
                not (self.param.head_outward and self.param.headrules):
            raise ValueError("ModifierGuide can only be used with head-outward binarization")


    def __init__(self, param: corpusparam):
        self._corpus = None
        self._grammar = None
        self.param = param
        self.check_parameters()
        self.tempdir = TemporaryDirectory()
        arc_filename = self._archive_filename()
        try:
            with tarfile.open(arc_filename, mode="r:gz") as arcfile:
                assert set(arcfile.getnames()) == set(self.__class__.ARCF), "damaged cache file"
                arcfile.extractall(self.tempdir.name)
        except FileNotFoundError:
            SupertagParseCorpus.extract_into(self.tempdir.name, self.param)
            makedirs(self.param.cachedir, exist_ok=True)
            with tarfile.open(arc_filename, mode="w:gz") as tarf:
                for fname in self.__class__.ARCF:
                    tarf.add(f"{self.tempdir.name}/{fname}", arcname=fname)


    @property
    def grammar(self):
        if self._grammar is None:
            core_supertags = set(
                token_to_supertag(token, supertag_attribs.from_str(self.param.core_attribs))
                for sentence in self.corpus.train
                for token in sentence
            )
            roots = set(
                next(token.get_tag("root_nonterminal").value for token in sentence if token.get_tag("root_nonterminal").value != "None")
                for sentence in self.corpus.train
            )
            self._grammar = SupertagGrammar(tuple(core_supertags), tuple(roots))
        return self._grammar


    @property
    def corpus(self):
        if self._corpus is None:
            self._corpus = SupertagParseCorpus(self.tempdir.name, supertag_attribs.from_str(self.param.core_attribs))
        return self._corpus


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.tempdir.cleanup()