from configparser import ConfigParser
from sys import argv
from train_flair import ParseCorpus, loadconfig
from tagger import Supertagger, hyperparam, evalparam
from flair.datasets.sequence_labeling import ColumnDataset
import logging
from timeit import default_timer
logging.getLogger("flair").setLevel(40)


if len(argv) < 3:
    print(f"use {argv[0]} <config> <model file> <values for k>+")
    exit(1)

config = ConfigParser()
config.read(argv[1])

lc = loadconfig(**config["Data"])

model = Supertagger.load(argv[2])
model.eval()
data = ParseCorpus(lc.corpusfilename)

for k in argv[3:]:
    print(f"running prediction for k = {k}")
    config["Test"]["ktags"] = k
    model.set_eval_param(evalparam(**config["Test"]))
    start = default_timer()
    results, _ = model.evaluate(data.dev, mini_batch_size=lc.batchsize, only_disc="both")
    end = default_timer()
    print(results.log_header)
    print(results.log_line)
    print(results.detailed_results)
    print()
    print(f"runtime: {end-start}s")
    print()