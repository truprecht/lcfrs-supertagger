from train_flair import read_config, loadconfig
from discodop.lexcorpus import SupertagCorpus
from functools import reduce

config = loadconfig(**read_config()["Data"])

corpus = SupertagCorpus.read(open(config.corpusfilename, "rb"))

size = len(corpus.sent_corpus)
splittype, *portions = config.split.split()
names = "train dev test".split()

assert splittype in ["hn", "vc", "dk"], (
    "split must be of the form `(hn|vc|dk) <train> <dev> <test>`, where"
    "hn rotates `test`, `dev` and `train` sentences for the test, developement and training set"
    "vc assumes ratios for `train`, `dev`, `test` and splits the sentences sequentially"
    "dk assumes absolute values and splits the sentences sequentially")

if splittype == "vc":
    portions = tuple(round(float(portion) * size) for portion in portions)
else:
    portions = tuple(int(portion) for portion in portions)

if splittype == "hn":
    mod = sum(portions)
    names = names[::-1]
    portions = portions[::-1]
else:
    mod = size

limits = tuple(reduce(lambda xs, y: xs + (xs[-1]+y,), portions, (0,)))
limits = tuple(zip(names, limits[:-1], limits[1:]))

files = { (name, tt): open(f"{config.corpusfilename}.{name}.{tt}", "w") for name in names for tt in ("trees", "tags") }

tdtidx = 0
for sentidx in range(size):
    while (sentidx%mod) < limits[tdtidx][1] or (sentidx%mod) >= limits[tdtidx][2]:
        tdtidx = (tdtidx+1) % len(limits)
    tagfile = files[(limits[tdtidx][0], "tags")]
    treefile = files[(limits[tdtidx][0], "trees")]
    for (word, pos, preterm, supertag) in zip(
            corpus.sent_corpus[sentidx],
            corpus.pos_corpus[sentidx],
            corpus.preterm_corpus[sentidx],
            corpus.supertag_corpus[sentidx]):
        print(f"{word} {corpus.pos[pos]} {corpus.preterms[preterm]} {corpus.supertags[supertag].pos()}", file=tagfile)
    print(file=tagfile)
    print(corpus.tree_corpus[sentidx], file=treefile)

for file in files.values(): file.close()