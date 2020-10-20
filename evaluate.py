from supertagging.data import SupertagParseCorpus, corpusparam
from supertagging.model import Supertagger, hyperparam, evalparam

from configparser import ConfigParser
from sys import argv
import logging
from timeit import default_timer
logging.getLogger("flair").setLevel(40)

from getopt import gnu_getopt
assert len(argv) >= 3, f"use {argv[0]} <data.conf> <model file> [options]"
opts, args = gnu_getopt(argv[1:], "", ("batchsize=", "ktags=", "only_disc=", "accuracy="))

config = ConfigParser()
config.read(args[0])
for option, v in opts:
    if not v: continue
    option = option[2:]
    config["Parsing"][option] = v

lc = corpusparam(**config["Corpus"], **config["Grammar"])
ec = evalparam(**config["Parsing"], batchsize=32)

model = Supertagger.load(args[1])
model.set_eval_param(ec)
model.eval()
data = SupertagParseCorpus(lc.filename)

start = default_timer()
results, _ = model.evaluate(data.test, mini_batch_size=ec.batchsize,
    only_disc=ec.only_disc, accuracy=ec.accuracy)
end = default_timer()
print(results.log_header)
print(results.log_line)
print(results.detailed_results)
print()
print(f"runtime: {end-start}s")