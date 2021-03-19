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
def main(config, name, args):
    from flair.trainers import ModelTrainer
    from flair.visual.training_curves import Plotter
    from math import ceil
    from torch.optim import Adam
    from torch import manual_seed
    from pickle import load
    from discodop.lexgrammar import SupertagGrammar

    cp = corpusparam(**config["Corpus"], **config["Grammar"])
    corpus = SupertagParseCorpus(cp.filename)
    if args.downsample:
        corpus = corpus.downsample(args.downsample)
    grammar = load(open(f"{cp.filename}.grammar", "rb"))

    tc = FindlrParameters(**config["Training"], **config["Eval-common"], **config["Eval-Development"], language=cp.language)
    model = Supertagger.from_corpus(corpus, grammar, tc)
    model.set_eval_param(tc)

    if args.iterations is None:
        epoch = ceil(len(corpus.train)/tc.batchsize)
        args.iterations = epoch * 5

    trainer = ModelTrainer(model, corpus)
    learning_rate_tsv = trainer.find_learning_rate(
        name,
        start_learning_rate=args.min_lr,
        end_learning_rate=args.max_lr,
        iterations=args.iterations)
    plotter = Plotter()
    plotter.plot_learning_rate(learning_rate_tsv)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from configparser import ConfigParser
    from os.path import basename
    from datetime import datetime

    args = ArgumentParser()
    args.add_argument("configs", type=str, nargs="+")
    args.add_argument("--min_lr", type=float, default=1e-7)
    args.add_argument("--max_lr", type=float, default=1e-2)
    args.add_argument("--iterations", type=int)
    args.add_argument("--downsample", type=float)
    args = args.parse_args()

    conf = ConfigParser()
    for config in args.configs:
        conf.read(config)

    conffilenames = (basename(f) for f in args.configs)
    filename = ("trained-"
                f"{'-'.join(conffilenames)}-"
                f"{datetime.now():%d-%m-%y-%H:%M}")

    main(conf, filename, args)