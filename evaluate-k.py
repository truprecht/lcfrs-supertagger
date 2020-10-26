from supertagging.data import SupertagParseDataset, corpusparam
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
ecdict = {**config["Eval-common"], **config["Eval-Development"]}
for option, v in opts:
    if not v: continue
    option = option[2:]
    ecdict[option] = v

lc = corpusparam(**config["Corpus"], **config["Grammar"])

model = Supertagger.load(args[1])
model.eval()
data = SupertagParseDataset(f"{lc.filename}.dev")

for k in args[2:]:
    print(f"running prediction for k = {k}")
    ecdict["ktags"] = k
    ep = evalparam(**ecdict)
    model.set_eval_param(ep)
    results, _ = model.evaluate(data, mini_batch_size=ep.batchsize, only_disc=ep.only_disc, accuracy="kbest")
    print(results.log_header)
    print(results.log_line)
    print(results.detailed_results)
    print()