from supertagging.data import SupertagParseDataset, corpusparam
from supertagging.model import Supertagger, hyperparam, evalparam

from configparser import ConfigParser
from sys import argv
import logging
logging.getLogger("flair").setLevel(40)

from getopt import gnu_getopt
assert len(argv) >= 3, f"use {argv[0]} <data.conf> <model file> [options]"
opts, args = gnu_getopt(argv[1:], "", ("batchsize=", "ktags=", "only_disc=", "accuracy="))

config = ConfigParser()
config.read(args[0])
ecdict = {**config["Eval-common"], **config["Eval-Test"]}
for option, v in opts:
    if not v: continue
    option = option[2:]
    ecdict[option] = v

lc = corpusparam(**config["Corpus"], **config["Grammar"])
ec = evalparam(**ecdict)

model = Supertagger.load(args[1])
model.set_eval_param(ec)
model.eval()
data = SupertagParseDataset(f"{lc.filename}.test")
results, _ = model.evaluate(data, mini_batch_size=ec.batchsize,
    only_disc=ec.only_disc, accuracy=ec.accuracy)
print(results.log_header)
print(results.log_line)
print(results.detailed_results)