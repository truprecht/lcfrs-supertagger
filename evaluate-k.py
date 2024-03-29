""" Reads the development set of a corpus and parses the sentences with
    different amounts of predicted supertags (`k`).
"""
import flair
import torch

from supertagging.data import SupertagCorpusFile, corpusparam
from supertagging.tagging.tagger_model import DecoderModel, EvalParameters

import logging
logging.getLogger("flair").setLevel(40)

from argparse import ArgumentParser
args = ArgumentParser()
args.add_argument("conf", nargs="+", help="configuration files containing at least the [Corpus] section")
args.add_argument("model", help="trained model file as it is saved by training.py")
args.add_argument("k", nargs="+", type=int, help="one or more values for k")
args.add_argument("-o", nargs="+", help="override options in the [Eval] section of the configuration", metavar="option=value")
args.add_argument("--device", type=torch.device, help="sets the torch device", choices=["cpu"]+[f"cuda:{n}" for n in range(torch.cuda.device_count())])
args = args.parse_args()

if args.device:
    flair.device = args.device

from configparser import ConfigParser
config = ConfigParser()
for f in args.conf:
    config.read(f)
ecdict = dict(config["Eval"]) if "Eval" in config else {}
for ov in (args.o or []):
    option, value = ov.split("=")
    ecdict[option.strip()] = value.strip()

lc = corpusparam(**config["Corpus"], **config["Grammar"])

model = DecoderModel.load(args.model)
model.eval()

with SupertagCorpusFile(lc) as cf:
    for k in args.k:
        print(f"running prediction for k = {k}")
        ecdict["ktags"] = k
        ep = EvalParameters(**ecdict)
        model.set_eval_param(ep)
        results = model.evaluate(cf.corpus.dev, mini_batch_size=ep.batchsize, only_disc=ep.only_disc, accuracy="kbest", othertag_accuracy=False, return_loss=False)
        if flair.__version__ < "0.9":
            results, _ = results
        print(results.log_header)
        print(results.log_line)
        print(results.detailed_results)
        print()