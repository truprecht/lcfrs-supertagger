""" Plot supertag prediction loss w.r.t. learning rate:
    trains the model with linearily increasing lr for few epochs and
    record the prediction loss vs gold supertags in the dev set.
"""

from supertagging.data import corpusparam, SupertagCorpusFile
from supertagging.model import EvalParameters
from supertagging.tagging.tagger_model import DecoderModel, DecoderModelParameters
from supertagging.parameters import Parameters

FindlrParameters = Parameters.merge(
        Parameters(batchsize=(int, 1), optimizer=(str, "Adam")),
        DecoderModelParameters, EvalParameters)
def main(config, name, args):
    from flair.trainers import ModelTrainer
    from flair.visual.training_curves import Plotter
    from math import ceil
    from pickle import load

    cp = corpusparam(**config["Corpus"], **config["Grammar"])
    with SupertagCorpusFile(cp) as cf:
        tc = FindlrParameters(**config["Training"], **config["Eval-common"], **config["Eval-Development"], language=cp.language)
        model = DecoderModel.from_corpus(cf.corpus, cf.grammar, tc)
        model.set_eval_param(tc)

        corpus = cf.corpus.downsample(args.downsample) if args.downsample else cf.corpus

        if args.iterations is None:
            epoch = ceil(len(corpus.train)/tc.batchsize)
            args.iterations = epoch * 5

        trainer = ModelTrainer(model, corpus)
        learning_rate_tsv = trainer.find_learning_rate(
            name,
            start_learning_rate=args.min_lr,
            end_learning_rate=args.max_lr,
            iterations=args.iterations,
            mini_batch_size=tc.batchsize)
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

    conffilenames = (basename(f).replace('.conf', '') for f in args.configs)
    filename = ("lr-"
                f"{'-'.join(conffilenames)}-"
                f"{datetime.now():%d-%m-%y-%H:%M}")

    main(conf, filename, args)