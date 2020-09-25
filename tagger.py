from parameters import Parameters

from flair.models import SequenceTagger
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings, OneHotEmbeddings, CharacterEmbeddings
from flair.nn import Model

from flair.datasets.sequence_labeling import Corpus
from flair.data import Dictionary, Label

import torch

from discodop.lexgrammar import SupertagGrammar

hyperparam = Parameters(
        pos_embedding_dim=(int, 10), dropout=(float, 0.1), lang=(str, None),
        lstm_layers=(int, 1), lstm_size=(int, 100), ktags=(int, 1), fallback_prob=(float, 0.0))
class Supertagger(Model):
    def __init__(self, embeddings, grammar, 
                lstm_size: int, lstm_layers: int, tags: Dictionary, dropout: float, ktags: int, fallback_prob: float):
        super(Supertagger, self).__init__()
        self.supertags = SequenceTagger(
            lstm_size, embeddings, tags, "supertag",
            rnn_layers=lstm_layers, dropout=dropout, use_crf=False, use_rnn=True
        )
        self.__grammar__ = grammar
        self.__grammar__.fallback_prob = fallback_prob
        self.ktags = ktags

    @property
    def fallback_prob(self):
        return self.__grammar__.fallback_prob

    @fallback_prob.setter
    def fallback_prob(self, value: float):
        self.__grammar__.fallback_prob = float

    @classmethod
    def from_corpus(cls, corpus: Corpus, grammar: SupertagGrammar, parameters: hyperparam):
        emb = StackedEmbeddings([
            FlairEmbeddings(f"{parameters.lang}-forward"),
            FlairEmbeddings(f"{parameters.lang}-backward"),
            WordEmbeddings(parameters.lang),
            OneHotEmbeddings(corpus, field="pos", embedding_length=parameters.pos_embedding_dim, min_freq=0)])
        tags = corpus.make_label_dictionary("supertag")
        return cls(emb, grammar,
            parameters.lstm_size, parameters.lstm_layers,
            tags, parameters.dropout, parameters.ktags, parameters.fallback_prob)

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
            probs = (-scores.softmax(dim=2).log()).numpy()
            tags = argpartition(probs, self.ktags, axis=2)[:, :, 0:self.ktags]
            weights = take_along_axis(probs, tags, 2)
            for sentence, senttags, sentweights in zip(batch, tags, weights):
                for token, ktags, kweights, in zip(sentence, senttags, sentweights):
                    str_ktags = tuple(self.supertags.tag_dictionary.get_item_for_index(t) for t in ktags)
                    token.add_tags_proba_dist(label_name, [Label(t, w) for t, w in zip(str_ktags, kweights)])
                    vtag = kweights.argmin()
                    token.add_tag_label(label_name, Label(str_ktags[vtag], kweights[vtag]))
            store_embeddings(batch, storage_mode=embedding_storage_mode)
            if return_loss:
                return loss

    def evaluate(self, sentences, mini_batch_size=32, num_workers=1, embedding_storage_mode="none", out_path=None):
        from flair.datasets import DataLoader
        from flair.training_utils import Result
        from discodop.lexcorpus import to_parse_tree
        from discodop.tree import ParentedTree
        from discodop.treetransforms import unbinarize, removefanoutmarkers
        from discodop.eval import Evaluator, readparam
        
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        eval_loss = 0
        i = 0
        batches = 0
        evaluator = Evaluator(readparam("proper.prm"))
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
                        (l.value, l.score)
                        for l in token.get_tags_proba_dist('predicted')])
                sent = [token.text for token in sentence]
                pos = [token.get_tag("pos").value for token in sentence]
                parses = self.__grammar__.parse(sent, pos, pos, predicted_tags, posmode=True)
                try:
                    parse = next(parses)
                    parse = to_parse_tree(parse)
                except StopIteration:
                    leaves = (f"({p} {i})" for p, i in zip(pos, range(len(sent))))
                    parse = ParentedTree(f"(NOPARSE {' '.join(leaves)})")
                gold = ParentedTree(sentence.get_labels("tree")[0].value)
                gold = unbinarize(removefanoutmarkers(gold))
                parse = unbinarize(removefanoutmarkers(parse))
                evaluator.add(i, gold, list(sent), parse, list(sent))
                i += 1
            batches += 1
        evlscores = { k: float_or_zero(v) for k,v in evaluator.acc.scores().items() }
        return (
            Result(evlscores["lf"], "Parsing fscore", f"fscore: {evlscores['lf']}", evaluator.summary()),
            eval_loss / batches
        )


    def _get_state_dict(self):
        return {
            "state": self.state_dict(),
            "embeddings": self.supertags.embeddings,
            "lstm_size": self.supertags.hidden_size,
            "lstm_layers": self.supertags.rnn_layers,
            "tags": self.supertags.tag_dictionary,
            "dropout": self.supertags.dropout.p,
            "grammar": self.__grammar__.todict(),
            "ktags": self.ktags,
            "fallback_prob": self.fallback_prob
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls(
            state["embeddings"],
            SupertagGrammar.fromdict(state["grammar"]),
            state["lstm_size"], state["lstm_layers"], state["tags"], state["dropout"],
            state["ktags"], state["fallback_prob"])
        model.load_state_dict(state["state"])
        return model


def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0
    except:
        return 0.0
