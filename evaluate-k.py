""" Reads the development set of a corpus and parses the sentences with
    different amounts of predicted supertags (`k`).
"""
import flair
import torch

from supertagging.data import SupertagParseDataset, corpusparam
from supertagging.model import Supertagger, EvalParameters


from configparser import ConfigParser
from sys import argv
import logging
logging.getLogger("flair").setLevel(40)

from argparse import ArgumentParser
args = ArgumentParser()
args.add_argument("data")
args.add_argument("model")
args.add_argument("k", nargs="+", type=int)
args.add_argument("--batchsize", type=int)
args.add_argument("--only_disc", choices=("true", "false", "both", "param"))
args.add_argument("--device", type=torch.device)
args = args.parse_args()

if args.device:
    flair.device = args.device

config = ConfigParser()
config.read(args.data)
ecdict = {**config["Eval-common"], **config["Eval-Development"]}
for option in ("batchsize", "only_disc"):
    if not vars(args)[option] is None:
        ecdict[option] = vars(args)[option]

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