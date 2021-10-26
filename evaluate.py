""" Reads the test set of a corpus, parses the sentences and reports the scores
    as specified in the data configuration.
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
args.add_argument("-o", nargs="+", help="override options in the [Eval] section of the configuration", metavar="option=value")
args.add_argument("--device", type=torch.device, help="sets the torch device", choices=["cpu"]+[f"cuda:{n}" for n in range(torch.cuda.device_count())])
args.add_argument("--dev", action="store_true", help="Evaluate on dev set instead of test set")
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
ec = EvalParameters(**ecdict)

lc = corpusparam(**config["Corpus"], **config["Grammar"])
with SupertagCorpusFile(lc) as cf:
    model = DecoderModel.load(args.model)
    model.set_eval_param(ec)
    model.eval()
    data = cf.corpus.dev if args.dev else cf.corpus.test
    results = model.evaluate(data, mini_batch_size=ec.batchsize,
        only_disc=ec.only_disc, accuracy=ec.accuracy, return_loss=False, othertag_accuracy=ec.othertag_accuracy)
    if flair.__version__ < "0.9":
        results, _ = results
    print(results.log_header)
    print(results.log_line)
    print(results.detailed_results)