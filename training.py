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


def main():
    config = read_config()

    torch_device = device("cuda" if cuda.is_available() else "cpu")
    print(f"running on device {torch_device}")
    try:
        start_state = load(config["Training"]["save_file"], map_location=torch_device)
    except:
        start_state = None

    (train, test, val), data = load_data(config["Data"], \
        tag_distance=int(config["Training"]["tag_distance"]))

    model = tagger(data.dims, tagger.Hyperparameters.from_dict(config["Training"])).double()
    model.to(torch_device)
    train_model(model, config["Training"], train, test, torch_device=torch_device, \
        report_loss=report_tensorboard_scalars, start_state=start_state, \
        report_histogram=report_tensorboard_histogram)

    data.evaluate(
        predictions(model, val, k=int(config["Val"]["top_tags"]), torch_device=torch_device),
        paramfilename=config["Val"]["eval_param"], fallback=float(config["Val"]["fallback"]))

def load_data(config, tag_distance=1):
    # setup corpus
    corpus = SupertagCorpus.read(open(config["corpus"], "rb"))
    if isfile(config["corpus"] + ".mat"):
        corpus.read_confusion_matrix(open(config["corpus"]+".mat", "rb"))
    vocabulary = set(word for sentence in corpus.sent_corpus for word in sentence)

    # setup word embeddings
    embedding = embedding_factory(config["word_embedding"])
    embedding = TruncatedEmbedding(embedding, vocabulary)

    data = SupertagDataset(corpus, embedding, tag_distance=tag_distance)

    # setup corpus split
    train, dev, test = split_data(config["split"], data)

    # setup data loaders
    shuffle = config.getboolean("shuffle")
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, collate_fn=data.collate_test, pin_memory=True)
    test_loader = DataLoader(dev, batch_size=batch_size, collate_fn=data.collate_test, pin_memory=True)
    val_loader = DataLoader(test, batch_size=batch_size, collate_fn=data.collate_val, pin_memory=True)

    data.truncate_supertags(train.indices)

    return (train_loader, test_loader, val_loader), data


def train_model(model, training_conf, train_data, test_data, torch_device=device("cpu"), report_loss=print, report_histogram=None, start_state=None):
    epochs = int(training_conf["epochs"])
    lr, momentum, alpha = \
        (float(training_conf[k]) for k in ("lr", "momentum", "loss_balance"))
    save_epochs = int(training_conf["save_epochs"]) if "save_epochs" in training_conf else 0
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    iteration = 0
    start_epoch = 0
    if start_state:
        model.load_state_dict(start_state["model"])
        opt.load_state_dict(start_state["optimizer"])
        start_epoch = start_state["epoch"]
        iteration = start_state["iteration"]

    for epoch in range(start_epoch+1, epochs+1):
        for sample in train_data:
            iteration += 1
            words, pos, prets, stags, lens = (t.to(torch_device) for t in sample)
            opt.zero_grad()
            l1, l2 = model.train_loss(model.forward((words, pos, lens)), (prets, stags))
            l = (1-alpha) * l1 + alpha * l2
            report_loss({ "loss/train/preterminals": l1.item(),
                     "loss/train/supertags": l2.item(),
                     "loss/train/combined": l.item() },
                     iteration )
            l.backward()
            opt.step()

        with no_grad():
            dl1, dl2, dl = tensor(0.0), tensor(0.0), tensor(0.0)
            positions = []
            for sample in test_data:
                words, pos, prets, stags, lens = (t.to(torch_device) for t in sample)
                (pr_prets, pr_stags) = model.forward((words, pos, lens))
                l1, l2 = model.test_loss((pr_prets, pr_stags), (prets, stags))
                dl1 += l1
                dl2 += l2
                dl += (1-alpha) * l1 + alpha * l2
                if report_histogram:
                    gold_positions = tagger.index_in_sorted(pr_stags, stags)
                    gold_positions[gold_positions == -1] = pr_stags.shape[2]
                    positions.append(gold_positions[gold_positions >= 0])
            dl1 /= len(test_data)
            dl2 /= len(test_data)
            dl /= len(test_data)

            report_loss({ "loss/test/preterminals": dl1.item(),
                     "loss/test/supertags": dl2.item(),
                     "loss/test/combined": dl.item() },
                     epoch )
            if positions and report_histogram:
                report_histogram({ "pos/test/supertags": cat(positions) }, epoch)

        if save_epochs and epoch % save_epochs == 0:
            save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "iteration": iteration
            }, training_conf["save_file"])

    return model


def predictions(model, val_data, k=1, torch_device=device("cpu")):
    model.eval()
    with no_grad():
        for sample in val_data:
            words, trees, wordembeddings, pos, lens = sample
            x = tuple(t.to(torch_device) for t in (wordembeddings, pos, lens))
            for sent, gold, (pos, preterms, supertags, weights) in zip(words, trees, model.predict(x, k)):
                yield sent, gold, pos, preterms, supertags, weights


writer = SummaryWriter()
def report_tensorboard_scalars(scores, iteration_or_epoch):
    for (key, score) in scores.items():
        writer.add_scalar(key, score, iteration_or_epoch)

def report_tensorboard_histogram(values, iteration_or_epoch):
    for (key, vals) in values.items():
        writer.add_histogram(key, vals, iteration_or_epoch)


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