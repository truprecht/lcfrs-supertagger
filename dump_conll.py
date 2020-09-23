from train_flair import read_config, loadconfig
from discodop.lexcorpus import SupertagCorpus

config = loadconfig(**read_config()["Data"])

corpus = SupertagCorpus.read(open(config.corpusfilename, "rb"))

size = len(corpus.sent_corpus)
splittype, absolute_or_ratio, *portions = config.split.split()
names = "training dev test".split()

if absolute_or_ratio == "ratio":
    portions = tuple(round(float(portion) * size) for portion in portions)
elif absolute_or_ratio == "absolute":
    portions = tuple(int(portion) for portion in portions)
else:
    raise NotImplementedError()

if splittype == "serial":
    assert sum(portions) <= size
    starts = (sum(portions[:0]), sum(portions[:1]), sum(portions[:2]))
    ends = (sum(portions[:1]), sum(portions[:2]), sum(portions[:3]))
    idxs = (range(start, end) for start, end in zip(starts, ends))
else:
    raise NotImplementedError()

for idx, name in zip(idxs, names):
    with open(config.corpusfilename + f".{name}", "w") as dump:
        for sentidx in idx:
            for (word, pos, preterm, supertag) in zip(corpus.sent_corpus[sentidx], corpus.pos_corpus[sentidx], corpus.preterm_corpus[sentidx], corpus.supertag_corpus[sentidx]):
                print(f"{word} {corpus.pos[pos]} {corpus.preterms[preterm]} {corpus.supertags[supertag].pos()}", file=dump)
            print(file=dump)