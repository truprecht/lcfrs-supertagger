""" Reads the development set of a corpus and parses the sentences with
    different amounts of predicted supertags (`k`).
"""
import flair
import torch

from supertagging.data import SupertagParseDataset, corpusparam
from supertagging.model import Supertagger, EvalParameters

import logging
logging.getLogger("flair").setLevel(40)

from argparse import ArgumentParser
args = ArgumentParser()
args.add_argument("data", help="configuration file containing at least the [Corpus] section")
args.add_argument("model", help="trained model file as it is saved by training.py")
args.add_argument("k", nargs="+", type=int, help="one or more values for k")
args.add_argument("-o", nargs="+", help="override options in the [Eval] section of the configuration", metavar="option=value")
args.add_argument("--device", type=torch.device, help="sets the torch device", choices=["cpu"]+[f"cuda:{n}" for n in range(torch.cuda.device_count())])
args = args.parse_args()

if args.device:
    flair.device = args.device

from configparser import ConfigParser
config = ConfigParser()
config.read(args.data)
ecdict = {**config["Eval-common"], **config["Eval-Development"]}
for ov in (args.o or []):
    option, value = ov.split("=")
    ecdict[option.strip()] = value.strip()

lc = corpusparam(**config["Corpus"], **config["Grammar"])

model = Supertagger.load(args.model)
model.eval()
data = SupertagParseDataset(f"{lc.filename}.dev")

for k in args.k:
    print(f"running prediction for k = {k}")
    ecdict["ktags"] = k
    ep = EvalParameters(**ecdict)
    model.set_eval_param(ep)
    results, _ = model.evaluate(data, mini_batch_size=ep.batchsize, only_disc=ep.only_disc, accuracy="kbest", pos_accuracy=False, return_loss=False)
    print(results.log_header)
    print(results.log_line)
    print(results.detailed_results)
    print()