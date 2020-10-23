from supertagging.data import SupertagParseCorpus, corpusparam
from supertagging.model import Supertagger, hyperparam, evalparam


from configparser import ConfigParser
from sys import argv
import logging
logging.getLogger("flair").setLevel(40)

from getopt import gnu_getopt
opts, args = gnu_getopt(argv[1:], "", ("batchsize=", "only_disc="))
assert len(args) > 2, f"use {argv[0]} <data.conf> <model file> <values for k>+ [options]"

config = ConfigParser()
config.read(args[0])
for option, v in opts:
    if not v: continue
    option = option[2:]
    config["Parsing"][option] = v

lc = corpusparam(**config["Corpus"], **config["Grammar"])

model = Supertagger.load(args[1])
model.eval()
data = SupertagParseCorpus(lc.filename)

for k in args[2:]:
    print(f"running prediction for k = {k}")
    config["Parsing"]["ktags"] = k
    ep = evalparam(**config["Parsing"])
    model.set_eval_param(ep)
    results, _ = model.evaluate(data.dev, mini_batch_size=ep.batchsize, only_disc=ep.only_disc, accuracy="all")
    print(results.log_header)
    print(results.log_line)
    print(results.detailed_results)
    print()