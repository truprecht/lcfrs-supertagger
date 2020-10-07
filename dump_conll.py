from train_flair import read_config, loadconfig
from discodop.lexcorpus import SupertagCorpus
from functools import reduce

config = loadconfig(**read_config()["Data"])

corpus = SupertagCorpus.read(open(config.corpusfilename, "rb"))

size = len(corpus.sent_corpus)
portions = config.split.split()
names = "train dev test".split()

portions = tuple(int(portion) for portion in portions)
assert len(portions) in [3,4]

limits = tuple(reduce(lambda xs, y: xs + (xs[-1]+y,), portions, (0,)))
limits = tuple(zip(names, limits[-4:-1], limits[-3:]))

print(limits, size)

tdtidx = 0
for (name, start, stop) in limits:
    tagfile = open(f"{config.corpusfilename}.{name}.tags", "w")
    treefile = open(f"{config.corpusfilename}.{name}.trees", "w")
    for sentidx in range(start, stop):
        for (word, pos, preterm, supertag) in zip(
                corpus.sent_corpus[sentidx],
                corpus.pos_corpus[sentidx],
                corpus.preterm_corpus[sentidx],
                corpus.supertag_corpus[sentidx]):
            print(f"{word} {corpus.pos[pos]} {corpus.preterms[preterm]} {corpus.supertags[supertag].pos()}", file=tagfile)
        print(file=tagfile)
        print(corpus.tree_corpus[sentidx], file=treefile)
    treefile.close()
    tagfile.close()