import flair
import torch

from supertagging.data import corpusparam, SupertagCorpusFile
from supertagging.model import EvalParameters
from supertagging.parameters import Parameters
from supertagging.tagging.tagger_model import DecoderModel, DecoderModelParameters

TrainingParameters = Parameters.merge(
        Parameters(epochs=(int, 1), lr=(float, 0.01), batchsize=(int, 1), weight_decay=(float, 0.01),
        patience=(int, 1), lr_decay=(float, 0.5), min_lr=(float, 0.0), optimizer=(str, "Adam")),
        DecoderModelParameters, EvalParameters)
def main(config, name, random_seed, param_selection_mode: bool = False):
    from flair.trainers import ModelTrainer
    from torch import manual_seed
    from pickle import load

    manual_seed(random_seed)

    cp = corpusparam(**config["Corpus"], **config["Grammar"])
    with SupertagCorpusFile(cp) as cf:
        tc = TrainingParameters(**config["Training"], **config["Eval-common"], **config["Eval-Development"], language=cp.language)
        model = DecoderModel.from_corpus(cf.corpus, cf.grammar, tc)
        model.set_eval_param(tc)

        trainer = ModelTrainer(model, cf.corpus, optimizer=getattr(torch.optim, tc.optimizer), use_tensorboard=True)
        trainer.train(
            name,
            learning_rate=tc.lr,
            mini_batch_size=tc.batchsize,
            max_epochs=tc.epochs,
            checkpoint=True,
            min_learning_rate=tc.min_lr,
            weight_decay=tc.weight_decay,
            patience=tc.patience,
            anneal_factor=tc.lr_decay,
            param_selection_mode=param_selection_mode)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from configparser import ConfigParser
    from os.path import basename
    from datetime import datetime

    args = ArgumentParser()
    args.add_argument("configs", nargs="+", help="configuration files for corpus and model, e.g. `example.conf example-model.conf`")
    args.add_argument("--seed", type=int, help="sets an integer as seed for initialization of the neural network, default is `0`", default=0)
    args.add_argument("--device", type=torch.device, choices=["cpu"]+[f"cuda:{n}" for n in range(torch.cuda.device_count())], help="sets the torch device")
    args = args.parse_args()

    if args.device:
        flair.device = args.device

    conf = ConfigParser()
    for config in args.configs:
        conf.read(config)

    conffilenames = (basename(f).replace('.conf', '') for f in args.configs)
    filename = ("trained-"
                f"{'-'.join(conffilenames)}-"
                f"{args.seed}-"
                f"{datetime.now():%d-%m-%y-%H:%M}")

    main(conf, filename, args.seed)