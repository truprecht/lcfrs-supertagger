""" Reads the traing set of a corpus and parses the sentences with supertags
    that are extracted from the gold parse trees. This should yield results
    close to 100% f-score and is solely used to estimate the impact of
    ambiguous parses with supertags.
"""

from sys import argv
from pickle import load
from configparser import ConfigParser
from supertagging.data import SupertagParseDataset
from supertagging.model import str_or_none

config = ConfigParser()
config.read(argv[1])

config = { **config["Corpus"], **config["Eval-common"], **config["Eval-Development"], **config["Grammar"] }
sep = tuple(config["separate_attribs"].split())
data = SupertagParseDataset(f"{config['filename']}.train", sep)

from discodop.tree import ParentedTree
from discodop.eval import Evaluator, readparam

grammar = load(open(f"{config['filename']}.grammar", "rb"))
print("using", len(grammar.tags), "supertags")
i = 0
evaluator = Evaluator(readparam(config["evalfilename"]))
for sentence in data:
    words = tuple(t.text for t in sentence)
    poss = tuple(t.get_tag("pos").value for t in sentence) if "pos" in sep else None
    constituents = tuple(str_or_none(t.get_tag("constituent").value) for t in sentence) if "constituent" in sep else None
    tags = tuple(((t.get_tag("supertag").value, 0.0),) for t in sentence)
    parses = grammar.parse(tags, str_tag_mode=True, pos=poss, constituent=constituents)
    try:
        parse = next(parses)
    except StopIteration:
        leaves = (f"({p} {i})" for p, i in zip(poss, range(len(words))))
        parse = ParentedTree(f"(NOPARSE {' '.join(leaves)})")
    gold = ParentedTree(sentence.get_labels("tree")[0].value)
    parse = ParentedTree.convert(parse)
    evaluator.add(i, gold.copy(deep=True), list(words), parse.copy(deep=True), list(words))
    i += 1
print(evaluator.summary())