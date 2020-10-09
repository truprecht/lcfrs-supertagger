from sys import argv
from train_flair import read_config
from pickle import load

c = read_config()

dataset = []
with open(f"{c['Data']['corpusfilename']}.test.tags") as file:
    words, poss, tags = [], [], []
    dataset = []
    for line in file:
        if not line.strip():
            dataset.append({
                "words": words,
                "poss": poss,
                "tags": tags})
            words, poss, tags = [], [], []
        else:
            word, pos, tag = line.strip().split()
            words.append(word)
            poss.append(pos)
            tags.append(tag)

with open(f"{c['Data']['corpusfilename']}.test.trees") as file:
    for line, record in zip(file, dataset):
        record["tree"] = line.strip()

from discodop.lexcorpus import to_parse_tree
from discodop.tree import ParentedTree, Tree
from discodop.treetransforms import unbinarize, removefanoutmarkers
from discodop.eval import Evaluator, readparam
from discodop.lexgrammar import SupertagGrammar

grammar = SupertagGrammar(load(open(c['Data']['corpusfilename'], "rb")))
i = 0
evaluator = Evaluator(readparam("proper.prm"))
for record in dataset:
    parses = grammar.parse(record["words"], record["poss"], [[(t, 0.0)] for t in record["tags"]], posmode=True)
    try:
        parse = next(parses)
        parse = to_parse_tree(parse)
    except StopIteration:
        leaves = (f"({p} {i})" for p, i in zip(record["poss"], range(len(record["words"]))))
        parse = ParentedTree(f"(NOPARSE {' '.join(leaves)})")
    gold = ParentedTree(record["tree"])
    gold = ParentedTree.convert(unbinarize(removefanoutmarkers(Tree.convert(gold))))
    parse = ParentedTree.convert(unbinarize(removefanoutmarkers(Tree.convert(parse))))
    evaluator.add(i, gold.copy(deep=True), list(record["words"]), parse.copy(deep=True), list(record["words"]))
    i += 1
print(evaluator.summary())