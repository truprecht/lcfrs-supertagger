from supertagging.data import corpusparam, SupertagParseCorpus
from supertagging.model import Supertagger, ModelParameters, EvalParameters
from supertagging.parameters import Parameters

TrainingParameters = Parameters.merge(
        Parameters(epochs=(int, 1), lr=(float, 0.01), batchsize=(int, 1), weight_decay=(float, 0.01)),
        ModelParameters, EvalParameters)
def main(config, name, random_seed):
    from flair.trainers import ModelTrainer
    from torch.optim import AdamW
    from torch import manual_seed
    from pickle import load
    from discodop.lexgrammar import SupertagGrammar

    manual_seed(random_seed)

    cp = corpusparam(**config["Corpus"], **config["Grammar"])
    corpus = SupertagParseCorpus(cp.filename)
    grammar = load(open(f"{cp.filename}.grammar", "rb"))

    tc = TrainingParameters(**config["Training"], **config["Eval-common"], **config["Eval-Development"], language=cp.language)
    model = Supertagger.from_corpus(corpus, grammar, tc)
    model.set_eval_param(tc)

    trainer = ModelTrainer(model, corpus, optimizer=AdamW, use_tensorboard=True)
    trainer.train(
        name,
        learning_rate=tc.lr,
        mini_batch_size=tc.batchsize,
        max_epochs=tc.epochs,
        checkpoint=True,
        min_learning_rate=tc.lr/4,
        weight_decay=tc.weight_decay)

if __name__ == "__main__":
    from sys import argv
    from configparser import ConfigParser
    from os.path import basename
    from datetime import datetime

    assert len(argv) in (3, 4), f"use {argv[0]} <data.conf> <model.conf> [random seed]"
    random_seed = int(argv[3]) if len(argv) == 4 else 0

    conf = ConfigParser()
    conf.read(argv[1])
    conf.read(argv[2])

    filename = ("trained-"
                f"{basename(argv[1]).replace('.conf', '')}-"
                f"{basename(argv[2]).replace('.conf', '')}-"
                f"{random_seed}-"
                f"{datetime.now():%d-%m-%y-%H:%M}")

    main(conf, filename, random_seed)