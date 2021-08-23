from supertagging.data import corpusparam

from functools import reduce
from pickle import dump
from configparser import ConfigParser
from sys import argv

assert len(argv) == 2, (f"use {argv[0]} <data.conf>")

cp = ConfigParser()
cp.read(argv[1])
config = corpusparam(**cp["Corpus"], **cp["Grammar"])

from discodop.treebank import READERS

from discodop.supertags.extraction import SupertagExtractor, GuideFactory, CompositionalNtConstructor
from discodop.supertags import SupertagGrammar

corpus = READERS[config.inputfmt](config.filename, encoding=config.inputenc, punct="move", headrules=config.headrules or None)
extract = SupertagExtractor(
    GuideFactory(config.guide),
    CompositionalNtConstructor(config.nonterminal_features.split()),
    separate_attribs=config.separate_attribs.split(),
    headoutward=bool(config.headrules),
    horzmarkov=config.h, vertmarkov=config.v)

portions = config.split.split()
names = "train dev test".split()
assert len(portions) in [3,4]

if portions[0] == "debug":
    portions = tuple(int(portion) for portion in portions[1:2]+portions[1:])
    limits = tuple((name, slice(0, end)) for name, end in zip(names, portions))
else:
    portions = tuple(int(portion) for portion in portions)
    limits = tuple(reduce(lambda xs, y: xs + (xs[-1]+y,), portions, (0,)))
    limits = tuple((name, slice(start, end)) for (name, start, end) in zip(names, limits[-4:-1], limits[-3:]))

tdtidx = 0
tagfiles = { name: open(f"{config.filename}.{name}.tags", "w") for name, _ in limits }
treefiles = { name: open(f"{config.filename}.{name}.trees", "w") for name, _ in limits }
for i, (_, item) in enumerate(corpus.itertrees()):
    subs = [name for name, s in limits if s.start <= i < s.stop]
    supertags = list(extract(item.tree, keep=("train" in subs)))
    for sub in subs:
        for word, tag in zip(item.sent, supertags):
            tag, *atts = tag
            print(word, tag.str_tag(), *atts, file=tagfiles[sub])
        print(file=tagfiles[sub])
        print(item.tree, file=treefiles[sub])
dump(SupertagGrammar(tuple(extract.supertags), tuple(extract.roots)), open(f"{config.filename}.grammar", "wb"))
for file in tagfiles.values():
    file.close()
for file in treefiles.values():
    file.close()