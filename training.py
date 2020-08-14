from discodop.lexcorpus import SupertagCorpus
from discodop.lexgrammar import SupertagGrammar
from discodop.tree import ParentedTree

from torch import cat, optim, save, load, no_grad, device, cuda, tensor
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, random_split

from os.path import isfile

from tagger import tagger
from dataset import SupertagDataset, embedding_factory, TruncatedEmbedding, split_data


def load_data(config, tag_distance=1):
    # setup word embeddings
    embedding = embedding_factory(config["Data"]["word_embedding"])

    # setup corpus
    corpus = SupertagCorpus.read(open(config["Data"]["corpus"], "rb"))
    if isfile(config["Data"]["corpus"] + ".mat"):
        corpus.read_confusion_matrix(open(config["Data"]["corpus"]+".mat", "rb"))
    vocabulary = set(word for sentence in corpus.sent_corpus for word in sentence)
    embedding = TruncatedEmbedding(embedding, vocabulary)

    grammar = SupertagGrammar(corpus)
    data = SupertagDataset(corpus, lambda w: embedding.stoi.get(w, -1), tag_distance=tag_distance)
    dims = data.dims

    # setup corpus split
    train, dev, test = split_data(config["Data"]["split"], data)
    training_tags = set(st.item() for _, _, _, _, _, sts in train for st in sts)

    # setup data loaders
    shuffle = config["Data"].getboolean("shuffle")
    batch_size = int(config["Data"]["batch_size"])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, collate_fn=data.collate_training)
    test_loader = DataLoader(dev, batch_size=batch_size, collate_fn=data.collate_test)
    val_loader = DataLoader(test, batch_size=batch_size, collate_fn=data.collate_val)
    
    return (train_loader, test_loader, val_loader), embedding, dims, training_tags, grammar


def train_model(training_conf, data_conf=None, torch_device=device("cpu"), report_loss=print, report_histogram=None, start_state=None):
    (training, test, val), embedding, dims, training_tags, grammar = load_data(data_conf, float(training_conf["tag_distance"]))
    epochs = int(training_conf["epochs"])
    lr, momentum, alpha = \
        (float(training_conf[k]) for k in ("lr", "momentum", "loss_balance"))
    save_epochs = int(training_conf["save_epochs"]) if "save_epochs" in training_conf else 0

    # setup Model
    model = tagger(dims, embedding.vectors, tagger.Hyperparameters.from_dict(training_conf))
    model.double()
    model.to(torch_device)
    ce_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
    softmax = LogSoftmax(dim=1)
    kl_loss = KLDivLoss(reduction='batchmean')
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    iteration = 0
    start_epoch = 0
    if start_state:
        model.load_state_dict(start_state["model"])
        opt.load_state_dict(start_state["optimizer"])
        start_epoch = start_state["epoch"]
        iteration = start_state["iteration"]

    for epoch in range(start_epoch+1, epochs+1):
        for sample in training:
            iteration += 1
            words, pos, prets, stags, lens = (t.to(torch_device) for t in sample)
            opt.zero_grad()
            (pr_prets, pr_stags) = model.forward((words, pos, lens))
            mask = model.get_mask(lens)
            l1, l2 = ce_loss(pr_prets[mask], prets[mask]), kl_loss(softmax(pr_stags[mask]), stags[mask])
            l = (1-alpha) * l1 + alpha * l2
            report_loss({ "loss/train/preterminals": l1.item(),
                     "loss/train/supertags": l2.item(),
                     "loss/train/combined": l.item() },
                     iteration )
            l.backward()
            opt.step()

        with no_grad():
            dl1, dl2 = 0.0, 0.0
            histograms = {
                "pos/test/supertags": [],
                "pos/test/supertags/trained": [],
                "pos/test/supertags/untrained": [],
                "score/test/supertags": [],
                "score/test/supertags/trained": [],
                "score/test/supertags/untrained": [] }
            for sample in test:
                words, pos, prets, stags, lens = sample
                words, pos, prets, stags, lens = (t.to(torch_device) for t in sample)
                (pr_prets, pr_stags) = model.forward((words, pos, lens))
                mask = model.get_mask(lens)
                l1, l2 = ce_loss(pr_prets[mask], prets[mask]), ce_loss(pr_stags[mask], stags[mask])
                dl1 += l1.item()
                dl2 += l2.item()
                if report_histogram:
                    gold_positions = tagger.index_in_sorted(pr_stags, stags)
                    gold_positions = gold_positions[gold_positions != -1]
                    gold_in_training = tensor([v.item() in training_tags for v in stags.flatten() if v.item() != -1])
                    histograms["pos/test/supertags"].append(gold_positions)
                    histograms["pos/test/supertags/trained"].append(gold_positions[gold_in_training])
                    histograms["pos/test/supertags/untrained"].append(gold_positions[~gold_in_training])
                    not_padded = (prets != -1)
                    gold_scores = pr_prets.gather(2, (prets * not_padded).unsqueeze(2))
                    gold_scores = gold_scores[not_padded]
                    histograms["score/test/supertags"].append(gold_scores)
                    histograms["score/test/supertags/trained"].append(gold_scores[gold_in_training])
                    histograms["score/test/supertags/untrained"].append(gold_scores[~gold_in_training])
            dl1 /= len(test)
            dl2 /= len(test)

            report_loss({ "loss/test/preterminals": dl1,
                     "loss/test/supertags": dl2,
                     "loss/test/combined": dl1+dl2 },
                     epoch )
            if report_histogram:
                report_histogram({ k: cat(values) for k, values in histograms.items() }, epoch)

        if save_epochs and epoch % save_epochs == 0:
            save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "iteration": iteration
            }, training_conf["save_file"])

    validate(model, val, grammar)


def first_or_noparse(derivations, sentence, pos):
    from discodop.lexcorpus import to_parse_tree
    try:
        deriv = next(derivations)
        deriv = to_parse_tree(deriv)
        return deriv
    except StopIteration:
        leaves = (f"({p} {i})" for p, i in zip(pos, range(len(sentence))))
        return ParentedTree(f"(NOPARSE {' '.join(leaves)})")

def unbin(parse):
    from discodop.treebanktransforms import removefanoutmarkers
    from discodop.treetransforms import unbinarize
    return unbinarize(removefanoutmarkers(parse))

def validate(model, val_data, grammar):
    from discodop.eval import Evaluator, readparam
    evaluator = Evaluator(readparam(config["Val"]["eval_param"]))
    model.eval()
    k = int(config["Val"]["top_tags"])
    with no_grad():
        i = 0
        for sample in val_data:
            words, trees, wordembeddings, pos, prets, stags, lens = sample
            (wordembeddings, pos, prets, stags, lens) = (t.to(torch_device) for t in (wordembeddings, pos, prets, stags, lens))
            (pret_scores, stag_scores) = model((wordembeddings, pos, lens))
            preterminals, supertags, weights = tagger.n_best_tags((pret_scores, stag_scores), k)
            for batch_idx, sequence_len in enumerate(lens):
                sequence_preterminals = preterminals[0:sequence_len, batch_idx].numpy()
                sequence_supertags = supertags[0:sequence_len, batch_idx]
                sequence_weights = weights[0:sequence_len, batch_idx]
                sequence_pos = pos[0:sequence_len, batch_idx].numpy()
                derivs = grammar.deintegerize_and_parse(words[batch_idx], sequence_pos, sequence_preterminals, sequence_supertags, sequence_weights, 1)
                deriv = first_or_noparse(derivs, words[batch_idx], [grammar.pos[n] for n in sequence_pos])
                evaluator.add(i, unbin(trees[batch_idx]), list(words[batch_idx]), unbin(deriv), list(words[batch_idx]))
                i += 1
        print(evaluator.summary())


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
    config = read_config()
    training_conf = config["Training"]
    torch_device = device("cuda" if cuda.is_available() else "cpu")
    print(f"running on device {torch_device}")
    try:
        start_state = load(config["Training"]["save_file"], map_location=torch_device)
    except:
        start_state = None
    train_model(training_conf, config, torch_device=torch_device, \
        report_loss=report_tensorboard_scalars, start_state=start_state, \
        report_histogram=report_tensorboard_histogram)