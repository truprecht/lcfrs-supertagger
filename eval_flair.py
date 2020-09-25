from configparser import ConfigParser
from sys import argv
from train_flair import ParseCorpus, loadconfig
from tagger import Supertagger, hyperparam, evalparam
from flair.datasets.sequence_labeling import ColumnDataset
import logging
logging.getLogger("flair").setLevel(40)


if len(argv) < 3:
    print(f"use {argv[0]} <config> [<corpus>] <model file>")
    exit(1)

config = ConfigParser()
config.read(argv[1])
if len(argv) > 3:
    config["Data"]["corpus"] = argv[2]

lc = loadconfig(**config["Data"])
ec = evalparam(**config["Test"])

model = Supertagger.load(argv[3] if len(argv) > 3 else argv[2])
model.set_eval_param(ec)
data = ParseCorpus(lc.corpusfilename)

model.eval()
results, _ = model.evaluate(data.test, mini_batch_size=lc.batchsize)
print(results.log_line)
print(results.detailed_results)