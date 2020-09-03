from discodop.lexcorpus import SupertagCorpus
from discodop.lexgrammar import SupertagGrammar
from discodop.tree import ParentedTree

from torch import cat, optim, save, load, no_grad, device, cuda, tensor, zeros
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, random_split

from os.path import isfile

from tagger import tagger
from dataset import SupertagDataset, embedding_factory, TruncatedEmbedding, split_data
from parameters import Parameters

testparam = Parameters(toptags=(int, 5), evalfilename=(str, None), fallback=(float, 0.0))
def main():
    config = read_config()

    torch_device = device("cuda" if cuda.is_available() else "cpu")
    tc = trainparam(**config["Training"], **config["Test"])
    print(f"running on device {torch_device}")
    try:
        start_state = load(tc.checkpoint_filename, map_location=torch_device)
    except:
        start_state = None

    (train, test, val), data = load_data(loadconfig(**config["Data"]))

    model = tagger(data.dims, tc).double().to(torch_device)
    train_model(model, tc, train, test, data, torch_device=torch_device, \
        report_loss=report_tensorboard_scalars, start_state=start_state)

    model.eval()
    ec = testparam(**config["Test"])
    data.evaluate(
        predictions(model, val, k=ec.toptags, torch_device=torch_device),
        paramfilename=ec.evalfilename, fallback=ec.fallback)

loadconfig = Parameters(
    corpusfilename=(str, None), wordembedding=(str, None),
    split=(str, None), batchsize=(int, 1), shuffle=(bool, False))
def load_data(config, vector_cache=None):
    # setup corpus
    corpus = SupertagCorpus.read(open(config.corpusfilename, "rb"))
    vocabulary = set(word for sentence in corpus.sent_corpus for word in sentence)

    # setup word embeddings
    embedding = embedding_factory(config.wordembedding, vector_cache)
    embedding = TruncatedEmbedding(embedding, vocabulary)

    data = SupertagDataset(corpus, embedding)

    # setup corpus split
    train, dev, test = split_data(config.split, data)

    # setup data loaders
    train_loader = DataLoader(train, batch_size=config.batchsize, shuffle=config.shuffle, collate_fn=data.collate_common, pin_memory=True)
    test_loader = DataLoader(dev, batch_size=config.batchsize, collate_fn=data.collate_common, pin_memory=True)
    val_loader = DataLoader(test, batch_size=config.batchsize, collate_fn=data.collate_common, pin_memory=True)

    data.truncate_supertags(train.indices)

    return (train_loader, test_loader, val_loader), data

trainparam = Parameters(
    epochs=(int, 1), checkpoint_epochs=(int, 0), checkpoint_filename=(str, None),
    lr=(float, 0.01), momentum=(float, 0.9),
    loss_balance=(float, 0.5))
trainparam = Parameters.merge(trainparam, testparam, tagger.hyperparam)
def train_model(model, config, train_data, test_data, data, torch_device=device("cpu"), report_loss=print, start_state=None):
    opt = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    iteration = 0
    start_epoch = 0
    if start_state:
        model.load_state_dict(start_state["model"])
        opt.load_state_dict(start_state["optimizer"])
        start_epoch = start_state["epoch"]
        iteration = start_state["iteration"]

    for epoch in range(start_epoch+1, config.epochs+1):
        for sample in train_data:
            iteration += 1
            sample = sample.to(torch_device)
            opt.zero_grad()
            l1, l2 = model.train_loss(model.forward(*sample.inp), sample.out)
            l = (1-config.loss_balance) * l1 + config.loss_balance * l2
            report_loss({ "loss/train/preterminals": l1.item(),
                     "loss/train/supertags": l2.item(),
                     "loss/train/combined": l.item() },
                     iteration )
            l.backward()
            opt.step()

        with no_grad():
            if not test_data: continue
            scores = data.evaluate(
                predictions(model, test_data, k=config.toptags, torch_device=torch_device),
                paramfilename=config.evalfilename, fallback=config.fallback)
            report_loss({
                    **{ "{}/test/parse".format(name): scores[name] for name in ("lf", "lp", "lr", "ex") },
                    "acc/test/supertags": scores["acc/tags"],
                    "acc/test/preterminals": scores["acc/preterms"],
                }, epoch )

        if config.checkpoint_epochs and epoch % config.checkpoint_epochs == 0:
            save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "iteration": iteration
            }, config.checkpoint_filename)

    return model


def predictions(model, val_data, k=1, torch_device=device("cpu")):
    for sample in val_data:
        sample = sample.to(torch_device)
        for golds, predictions \
                in zip(sample.golds(), model.predict(*sample.inp, k)):
            yield (*golds, *predictions)


def report_tensorboard_scalars(scores, iteration_or_epoch, writer=SummaryWriter()):
    for (key, score) in scores.items():
        writer.add_scalar(key, score, iteration_or_epoch)


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