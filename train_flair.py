from parameters import Parameters

from discodop.lexgrammar import SupertagCorpus, SupertagGrammar
from tagger import Supertagger, hyperparam, evalparam
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.trainers import ModelTrainer

from torch.optim import Adam

class ParseCorpus(ColumnCorpus):
    def __init__(self, basename):
        super(ParseCorpus, self).__init__(
            "", {0: "text", 1: "pos", 2: "preterm", 3: "supertag"},
            train_file=f"{basename}.train.tags", test_file=f"{basename}.test.tags", dev_file=f"{basename}.dev.tags")
        for s in  ("train", "test", "dev"):
            with open(f"{basename}.{s}.trees") as file:
                for idx, tree in enumerate(file):
                    tree = tree.strip()
                    self.__getattribute__(s)[idx].add_label("tree", tree)

trainparam = Parameters(
    epochs=(int, 1), checkpoint_filename=(str, None),
    lr=(float, 0.01))
trainparam = Parameters.merge(trainparam, evalparam, hyperparam)
loadconfig = Parameters(
    corpusfilename=(str, None), split=(str, None), batchsize=(int, 1))
def main():
    config = read_config()

    dc = loadconfig(**config["Data"])
    corpus = ParseCorpus(dc.corpusfilename)
    grammar = SupertagGrammar(SupertagCorpus.read(open(dc.corpusfilename, "rb")), 0.0)

    tc = trainparam(**config["Training"], **config["Test"])
    model = Supertagger.from_corpus(corpus, grammar, tc)
    model.set_eval_param(tc)

    trainer = ModelTrainer(model, corpus, optimizer=Adam, use_tensorboard=True)
    trainer.train(
        tc.checkpoint_filename,
        learning_rate=tc.lr,
        mini_batch_size=dc.batchsize,
        max_epochs=tc.epochs,
        checkpoint=True,
        min_learning_rate=tc.lr*1e-3)

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