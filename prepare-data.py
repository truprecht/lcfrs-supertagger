from supertagging.data import corpusparam

from functools import reduce
from pickle import dump
from configparser import ConfigParser
from sys import argv

assert len(argv) == 2, (f"use {argv[0]} <data.conf>")

cp = ConfigParser()
cp.read(argv[1])
config = corpusparam(**cp["Corpus"], **cp["Grammar"], **cp["Lexicalization"])

from discodop.tree import Tree
from discodop.treebank import Item, READERS
from discodop.treetransforms import addfanoutmarkers, binarize, collapseunary
from discodop.lexgrammar import SupertagCorpus, SupertagGrammar

def add_bintree(corpus_item: Item, **args):
    # TODO: move this into Supertagcorpus
    bt = addfanoutmarkers(
     binarize(
      collapseunary(
       Tree.convert(corpus_item.tree), collapseroot=True, collapsepos=True),
      horzmarkov=config.h, vertmarkov=config.v, **args))
    return (corpus_item, bt)

corpus = READERS[config.inputfmt](config.filename, encoding=config.inputenc, punct="move", headrules=config.headrules or None)
corpus = (add_bintree(t, headoutward=bool(config.headrules)) for _, t in corpus.itertrees())

corpus = SupertagCorpus(
    corpus,
    split_strictness=config.split_strictness,
    marker=config.propterm_marker,
    renaming_scheme=config.propterm_nonterminals)

size = len(corpus.sent_corpus)
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
nt, strict = corpus.options.renaming_scheme, corpus.options.split_strictness
for (name, indices) in limits:
    tagfile = open(f"{config.filename}.{name}.tags", "w")
    treefile = open(f"{config.filename}.{name}.trees", "w")
    subcorpus = corpus.subcorpus(indices)
    if name == "train":
        dump(SupertagGrammar(subcorpus), open(f"{config.filename}.grammar", "wb"))
    for (sentence, poss, tags, tree) in zip(subcorpus.sent_corpus,
            subcorpus.pos_corpus, subcorpus.supertag_corpus, subcorpus.tree_corpus):
        for (word, pos, tag) in zip(sentence, poss, tags):
            print(f"{word} {subcorpus.pos[pos]} {subcorpus.supertags[tag].pos(nt, strict)}", file=tagfile)
        print(file=tagfile)
        print(tree, file=treefile)
    treefile.close()
    tagfile.close()