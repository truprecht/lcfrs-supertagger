from sys import argv
from pickle import load
from configparser import ConfigParser
from supertagging.data import SupertagParseCorpus

config = ConfigParser()
config.read(argv[1])

data = SupertagParseCorpus(config['Corpus']['filename'])

from discodop.tree import ParentedTree, Tree
from discodop.treetransforms import unbinarize, removefanoutmarkers
from discodop.eval import Evaluator, readparam
from discodop.lexgrammar import SupertagGrammar

grammar = SupertagGrammar(load(open(f"{config['Corpus']['filename']}.corpus", "rb")))
i = 0
evaluator = Evaluator(readparam("proper.prm"))
for sentence in data.test:
    words = tuple(t.text for t in sentence)
    poss = tuple(t.get_tag("pos").value for t in sentence)
    tags = tuple(((t.get_tag("supertag").value, 0.0),) for t in sentence)
    parses = grammar.parse(poss, tags, posmode=True)
    try:
        parse = next(parses)
    except StopIteration:
        leaves = (f"({p} {i})" for p, i in zip(poss, range(len(words))))
        parse = ParentedTree(f"(NOPARSE {' '.join(leaves)})")
    gold = ParentedTree(sentence.get_labels("tree")[0].value)
    gold = ParentedTree.convert(unbinarize(removefanoutmarkers(Tree.convert(gold))))
    parse = ParentedTree.convert(unbinarize(removefanoutmarkers(Tree.convert(parse))))
    evaluator.add(i, gold.copy(deep=True), list(words), parse.copy(deep=True), list(words))
    i += 1
print(evaluator.summary())