from discodop.lexcorpus import SupertagCorpus
from discodop.lexgrammar import SupertagGrammar
from discodop.tree import ParentedTree

from torch import cat, optim, save, load, no_grad, device, cuda, tensor, zeros
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, random_split

from os.path import isfile

from tagger import CrfTagger
from dataset import SupertagDataset, embedding_factory, TruncatedEmbedding, split_data
from parameters import Parameters

from tagger import CrfTagger, hyperparam
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.trainers import ModelTrainer
import numpy as np

DEBUG = True

testparam = Parameters(toptags=(int, 5), evalfilename=(str, None), fallback=(float, 0.0))
loadconfig = Parameters(
    corpusfilename=(str, None), wordembedding=(str, None),
    split=(str, None), batchsize=(int, 1))
def main():
    config = read_config()

    torch_device = device("cuda" if cuda.is_available() else "cpu")
    print(f"running on device {torch_device}")

    dc = loadconfig(**config["Data"])
    corpus = ColumnCorpus("", {0: "text", 1: "pos", 2: "preterm", 3: "supertag"}, \
        train_file=f"{dc.corpusfilename}.training", test_file=f"{dc.corpusfilename}.test", dev_file=f"{dc.corpusfilename}.dev")

    tc = trainparam(**config["Training"], **config["Test"])
    model = CrfTagger.from_corpus(corpus, tc)

    trainer = ModelTrainer(model, corpus, use_tensorboard=True)
    trainer.train(
        tc.checkpoint_filename,
        learning_rate=tc.lr,
        mini_batch_size=dc.batchsize,
        max_epochs=tc.epochs)

trainparam = Parameters(
    epochs=(int, 1), checkpoint_epochs=(int, 0), checkpoint_filename=(str, None),
    lr=(float, 0.01), momentum=(float, 0.9),
    loss_balance=(float, 0.5))
trainparam = Parameters.merge(trainparam, testparam, hyperparam)





def read_config():
    from sys import argv
    from configparser import ConfigParser

    if len(argv) < 2:
        print(f"use {argv[0]} <config> [<corpus>]")
        exit(1)

    config = ConfigParser()
    config.read(argv[1])
    if len(argv) > 2:
        config["Data"]["corpus"] = argv[2]

    return config


if __name__ == "__main__":
    main()