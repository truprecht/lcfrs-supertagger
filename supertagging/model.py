from .parameters import Parameters

from flair.datasets.sequence_labeling import Corpus
from flair.data import Dictionary, Label
from flair.models import SequenceTagger
from flair.nn import Model
from flair.training_utils import Result

import torch

from discodop.lexgrammar import SupertagGrammar
from discodop.eval import Evaluator, readparam

from typing import Tuple, Union


def pretrainedstr(model, language):
    if model != "bert-base":
        return model
    # translate two letter language code into bert model
    return {
        "de": "bert-base-german-cased",
        "en": "bert-base-cased",
        "nl": "wietsedv/bert-base-dutch-cased"
    }[language]

def EmbeddingFactory(parameters, corpus):
    from flair.embeddings import FlairEmbeddings, StackedEmbeddings, \
        WordEmbeddings, OneHotEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings

    stack = []
    for emb in parameters.embedding.split():
        if any((spec in emb) for spec in ("bert", "gpt", "xlnet")):
            stack.append(TransformerWordEmbeddings(
                model=pretrainedstr(emb, parameters.language),
                fine_tune=True))
        elif emb == "flair":
            stack += [
                FlairEmbeddings(f"{parameters.language}-forward", fine_tune=True, with_whitespace=False),
                FlairEmbeddings(f"{parameters.language}-backward", fine_tune=True, with_whitespace=False)]
        elif emb == "pos":
            stack.append(OneHotEmbeddings(corpus, field="pos", embedding_length=parameters.pos_embedding_dim, min_freq=0))
        elif emb == "word":
            stack.append(WordEmbeddings(parameters.language))
        else:
            raise NotImplementedError()
    return StackedEmbeddings(stack)

hyperparam = Parameters(
        pos_embedding_dim=(int, 10), dropout=(float, 0.1), language=(str, ""),
        lstm_layers=(int, 1), lstm_size=(int, 100), embedding=(str, "pos"))
evalparam = Parameters(
    ktags=(int, 5), fallbackprob=(float, 0.0),
    batchsize=(int, 1),
    evalfilename=(str, None), only_disc=(str, "both"), accuracy=(str, "all"))
class Supertagger(Model):
    def __init__(self, embeddings, grammar, 
                lstm_size: int, lstm_layers: int, tags: Dictionary, dropout: float):
        super(Supertagger, self).__init__()
        self.supertags = SequenceTagger(
            lstm_size, embeddings, tags, "supertag",
            rnn_layers=lstm_layers, dropout=dropout, use_crf=False, use_rnn=True, reproject_embeddings=False
        )
        self.__grammar__ = grammar
        self.__evalparam__ = None
        self.ktags = 1

    def set_eval_param(self, config):
        self.fallback_prob = config.fallbackprob
        self.evalparam = config.evalfilename
        self.ktags = config.ktags

    @property
    def fallback_prob(self):
        return self.__grammar__.fallback_prob

    @fallback_prob.setter
    def fallback_prob(self, value: float):
        self.__grammar__.fallback_prob = value

    @property
    def evalparam(self):
        return self.__evalparam__

    @evalparam.setter
    def evalparam(self, evalfile):
        self.__evalparam__ = readparam(evalfile)

    @classmethod
    def from_corpus(cls, corpus: Corpus, grammar: SupertagGrammar, parameters: hyperparam):
        tags = corpus.make_label_dictionary("supertag")
        return cls(
            EmbeddingFactory(parameters, corpus),
            grammar,
            parameters.lstm_size, parameters.lstm_layers,
            tags, parameters.dropout)

    def forward_loss(self, data_points):
        return self.supertags.forward_loss(data_points)

    def forward(self, data_points):
        return self.supertags.forward(data_points)

    def predict(self, batch, label_name=None, return_loss=False, embedding_storage_mode="none"):
        from flair.training_utils import store_embeddings
        from numpy import argpartition, take_along_axis
        if not label_name:
            label_name = self.supertags.tag_type
        with torch.no_grad():
            scores = self.supertags.forward(batch)
            loss = self.supertags._calculate_loss(scores, batch)
            probs = scores.softmax(dim=2).cpu().numpy()
            tags = argpartition(-probs, self.ktags, axis=2)[:, :, 0:self.ktags]
            weights = take_along_axis(probs, tags, 2)
            for sentence, senttags, sentweights in zip(batch, tags, weights):
                for token, ktags, kweights, in zip(sentence, senttags, sentweights):
                    str_ktags = tuple(self.supertags.tag_dictionary.get_item_for_index(t) for t in ktags)
                    token.add_tags_proba_dist(label_name, [Label(t, w) for t, w in zip(str_ktags, kweights)])
                    vtag = kweights.argmax()
                    token.add_tag_label(label_name, Label(str_ktags[vtag], kweights[vtag]))
            store_embeddings(batch, storage_mode=embedding_storage_mode)
            if return_loss:
                return loss

    def evaluate(self, sentences, mini_batch_size=32, num_workers=1,
            embedding_storage_mode="none", out_path=None,
            only_disc: str = "param", accuracy: str = "first") -> Tuple[Result, float]:
        """ :param sentences: a sentence ``DataSet`` of sentences, where
            each contains a label `tree` whose label is the gold parse tree, as
            provided by ``ParseCorpus``.
            :param disc_only: If set, overrides the setting `DISC_ONLY` in the
            evaluation parameter file ``self.evalparam``, i.e. only evaluates
            discontinuous constituents if True. Pass "both" to report both
            results.
            :param accuracy: either 'first' or 'all'. Determines if the
            accuracy is computed from the best, or k-best predicted tags.
            :returns: tuple with evaluation ``Result``, where the main score
            is the f1-score (for all constituents, if only_disc == "both").
        """
        from flair.datasets import DataLoader
        from discodop.tree import ParentedTree, Tree
        from discodop.treetransforms import unbinarize, removefanoutmarkers
        from discodop.eval import Evaluator, readparam
        from math import log

        if self.__evalparam__ is None:
            raise Exception("Need to specify evaluator parameter file before evaluating")
        if only_disc == "both":
            evaluators = {
                "all":  Evaluator({ **self.evalparam, "DISC_ONLY": False }),
                "disc": Evaluator({ **self.evalparam, "DISC_ONLY": True  })}
        else:
            mode = self.evalparam["DISC_ONLY"] if only_disc == "param" else (only_disc=="true")
            strmode = "disc" if mode else "all"
            evaluators = {
                strmode: Evaluator({ **self.evalparam, "DISC_ONLY": mode })}
        
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        eval_loss = 0
        i = 0
        batches = 0
        corr_tags = 0
        all_tags = 0
        for batch in data_loader:
            # predict for batch
            loss = self.predict(batch,
                    embedding_storage_mode=embedding_storage_mode,
                    label_name='predicted',
                    return_loss=True)
            eval_loss += loss

            for sentence in batch:
                predicted_tags = []
                for token in sentence:
                    predicted_tags.append([
                        (l.value, -log(l.score))
                        for l in token.get_tags_proba_dist('predicted')])
                    taglist = [l.value for l in token.get_tags_proba_dist('predicted')]
                    if accuracy == "all" and token.get_tag("supertag").value in taglist:
                        corr_tags += 1
                    elif accuracy == "first":
                        corr_tags += int(token.get_tag("supertag").value == token.get_tag('predicted').value)
                all_tags += len(sentence)
                sent = [token.text for token in sentence]
                pos = [token.get_tag("pos").value for token in sentence]
                parses = self.__grammar__.parse(sent, pos, predicted_tags, posmode=True)
                try:
                    parse = next(parses)
                except StopIteration:
                    leaves = (f"({p} {i})" for p, i in zip(pos, range(len(sent))))
                    parse = Tree(f"(NOPARSE {' '.join(leaves)})")
                gold = Tree(sentence.get_labels("tree")[0].value)
                gold = ParentedTree.convert(unbinarize(removefanoutmarkers(gold)))
                parse = ParentedTree.convert(unbinarize(removefanoutmarkers(parse)))
                for evaluator in evaluators.values():
                    evaluator.add(i, gold.copy(deep=True), list(sent), parse.copy(deep=True), list(sent))
                i += 1
            batches += 1
        scores = {
            strmode: float_or_zero(evaluator.acc.scores()['lf'])
            for strmode, evaluator in evaluators.items()}
        scores["accuracy"] = corr_tags / all_tags
        return (
            Result(
                scores['all'] if 'all' in scores else scores['disc'],
                "\t".join(f"fscore ({mode})" for mode in scores),
                "\t".join(f"{s}" for s in scores.values()),
                '\n\n'.join(evaluator.summary() for evaluator in evaluators.values())),
            eval_loss / batches
        )


    def _get_state_dict(self):
        return {
            "state": self.state_dict(),
            "embeddings": self.supertags.embeddings,
            "lstm_size": self.supertags.hidden_size,
            "lstm_layers": self.supertags.rnn_layers,
            "tags": self.supertags.tag_dictionary,
            "dropout": self.supertags.use_dropout,
            "grammar": self.__grammar__.todict(),
            "ktags": self.ktags,
            "evalparam": self.__evalparam__
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls(
            state["embeddings"],
            SupertagGrammar.fromdict(state["grammar"]),
            state["lstm_size"], state["lstm_layers"], state["tags"], state["dropout"])
        model.__evalparam__ = state["evalparam"]
        model.ktags = state["ktags"]
        model.load_state_dict(state["state"])
        return model


def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0
    except:
        return 0.0
