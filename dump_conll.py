from train_flair import read_config, loadconfig
from functools import reduce
from pickle import load

config = loadconfig(**read_config()["Data"])

corpus = load(open(config.corpusfilename, "rb"))

size = len(corpus.sent_corpus)
portions = config.split.split()
names = "train dev test".split()
assert len(portions) in [3,4]

if portions[0] == "debug":
    portions = tuple(int(portion) for portion in portions[1:2]+portions[1:])
    limits = tuple((name, 0, end) for name, end in zip(names, portions))
else:
    portions = tuple(int(portion) for portion in portions)
    limits = tuple(reduce(lambda xs, y: xs + (xs[-1]+y,), portions, (0,)))
    limits = tuple(zip(names, limits[-4:-1], limits[-3:]))

tdtidx = 0
for (name, start, stop) in limits:
    tagfile = open(f"{config.corpusfilename}.{name}.tags", "w")
    treefile = open(f"{config.corpusfilename}.{name}.trees", "w")
    for sentidx in range(start, stop):
        for (word, pos, supertag) in zip(
                corpus.sent_corpus[sentidx],
                corpus.pos_corpus[sentidx],
                corpus.supertag_corpus[sentidx]):
            print(f"{word} {corpus.pos[pos]} {corpus.supertags[supertag].pos()}", file=tagfile)
        print(file=tagfile)
        print(corpus.tree_corpus[sentidx], file=treefile)
    treefile.close()
    tagfile.close()