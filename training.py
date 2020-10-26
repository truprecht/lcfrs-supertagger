from supertagging.data import corpusparam, SupertagParseCorpus
from supertagging.model import Supertagger, hyperparam, evalparam
from supertagging.parameters import Parameters

trainparam = Parameters(
    epochs=(int, 1), lr=(float, 0.01), batchsize=(int, 1))
trainparam = Parameters.merge(trainparam, evalparam, hyperparam)
def main(config, name):
    from flair.trainers import ModelTrainer
    from torch.optim import Adam
    from pickle import load
    from discodop.lexgrammar import SupertagGrammar

    cp = corpusparam(**config["Corpus"], **config["Grammar"])
    corpus = SupertagParseCorpus(cp.filename)
    grammar = load(open(f"{cp.filename}.grammar", "rb"))

    tc = trainparam(**config["Training"], **config["Eval-common"], **config["Eval-Development"], language=cp.language)
    model = Supertagger.from_corpus(corpus, grammar, tc)
    model.set_eval_param(tc)

    trainer = ModelTrainer(model, corpus, optimizer=Adam, use_tensorboard=True)
    trainer.train(
        name,
        learning_rate=tc.lr,
        mini_batch_size=tc.batchsize,
        max_epochs=tc.epochs,
        checkpoint=True,
        min_learning_rate=tc.lr*1e-3)

if __name__ == "__main__":
    from sys import argv
    from configparser import ConfigParser
    from os.path import basename
    from datetime import datetime

    assert len(argv) == 3, f"use {argv[0]} <data.conf> <model.conf>"
    
    conf = ConfigParser()
    conf.read(argv[1])
    conf.read(argv[2])

    filename = ("trained-"
                f"{basename(argv[1]).replace('.conf', '')}-"
                f"{basename(argv[2]).replace('.conf', '')}-"
                f"{datetime.now(): %d-%m-%y-%H:%M}")

    main(conf, filename)