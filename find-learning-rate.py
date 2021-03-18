""" Plot supertag prediction loss w.r.t. learning rate:
    trains the model with linearily increasing lr for few epochs and
    record the prediction loss vs gold supertags in the dev set.
"""

from supertagging.data import corpusparam, SupertagParseCorpus
from supertagging.model import Supertagger, ModelParameters, EvalParameters
from supertagging.parameters import Parameters

FindlrParameters = Parameters.merge(
        Parameters(batchsize=(int, 1)),
        ModelParameters, EvalParameters)
def main(config, name, min, max, its: int = None):
    from flair.trainers import ModelTrainer
    from flair.visual.training_curves import Plotter
    from math import ceil
    from torch.optim import Adam
    from torch import manual_seed
    from pickle import load
    from discodop.lexgrammar import SupertagGrammar

    cp = corpusparam(**config["Corpus"], **config["Grammar"])
    corpus = SupertagParseCorpus(cp.filename)
    grammar = load(open(f"{cp.filename}.grammar", "rb"))

    tc = FindlrParameters(**config["Training"], **config["Eval-common"], **config["Eval-Development"], language=cp.language)
    model = Supertagger.from_corpus(corpus, grammar, tc)
    model.set_eval_param(tc)

    if its is None:
        epoch = ceil(len(corpus.train)/tc.batchsize)
        its = epoch * 5

    trainer = ModelTrainer(model, corpus, optimizer=Adam)
    learning_rate_tsv = trainer.find_learning_rate(name, start_learning_rate=min, end_learning_rate=max, iterations=its)
    plotter = Plotter()
    plotter.plot_learning_rate(learning_rate_tsv)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from configparser import ConfigParser
    from os.path import basename
    from datetime import datetime

    args = ArgumentParser()
    args.add_argument("data", type=str)
    args.add_argument("model", type=str)
    args.add_argument("--min_lr", type=float, default=1e-7)
    args.add_argument("--max_lr", type=float, default=1e-2)
    args.add_argument("--iterations", type=int)
    args = args.parse_args()

    conf = ConfigParser()
    conf.read(args.data)
    conf.read(args.model)

    filename = ("trained-"
                f"{basename(args.data).replace('.conf', '')}-"
                f"{basename(args.model).replace('.conf', '')}-"
                f"{datetime.now():%d-%m-%y-%H:%M}")

    main(conf, filename, args.min_lr, args.max_lr, args.iterations)