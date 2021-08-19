from typing import Tuple, Union, List

from flair.datasets.sequence_labeling import Corpus
from flair.data import Dictionary, Label, Sentence
from flair.nn import Model
from flair.training_utils import Result

import torch

from discodop.supertags import SupertagGrammar
from discodop.eval import Evaluator, readparam

from .data import SupertagParseDataset
from .parameters import Parameters
from .sequence_multi_tagger import SequenceMultiTagger


def pretrainedstr(model, language):
    if model != "bert-base":
        return model
    # translate two letter language code into bert model
    return {
        "de": "bert-base-german-cased",
        "en": "bert-base-cased",
        "nl": "wietsedv/bert-base-dutch-cased"
    }[language]


EmbeddingParameters = Parameters(
    embedding=(str, "word char"), tune_embedding=(bool, False), language=(str, ""),
    pos_embedding_dim=(int, 20),
    word_embedding_dim=(int, 300), word_minfreq=(int, 1),
    char_embedding_dim=(int, 64), char_bilstm_dim=(int, 100))
def EmbeddingFactory(parameters, corpus):
    from flair.embeddings import FlairEmbeddings, StackedEmbeddings, \
        WordEmbeddings, OneHotEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings

    stack = []
    for emb in parameters.embedding.split():
        if any((spec in emb) for spec in ("bert", "gpt", "xlnet")):
            stack.append(TransformerWordEmbeddings(
                model=pretrainedstr(emb, parameters.language),
                fine_tune=parameters.tune_embedding))
        elif emb == "flair":
            stack += [
                FlairEmbeddings(f"{parameters.language}-forward", fine_tune=parameters.tune_embedding),
                FlairEmbeddings(f"{parameters.language}-backward", fine_tune=parameters.tune_embedding)]
        elif emb == "pos":
            stack.append(OneHotEmbeddings(
                corpus,
                field="pos",
                embedding_length=parameters.pos_embedding_dim,
                min_freq=1))
        elif emb == "fasttext":
            stack.append(WordEmbeddings(parameters.language))
        elif emb == "word":
            stack.append(OneHotEmbeddings(
                corpus,
                field="text",
                embedding_length=parameters.word_embedding_dim,
                min_freq=parameters.word_minfreq))
        elif emb == "char":
            stack.append(CharacterEmbeddings(
                char_embedding_dim=parameters.char_embedding_dim,
                hidden_size_char=parameters.char_bilstm_dim))
        else:
            raise NotImplementedError()
    return StackedEmbeddings(stack)

ModelParameters = Parameters.merge(
        Parameters(
            dropout=(float, 0.0), word_dropout=(float, 0.0), locked_dropout=(float, 0.0), lstm_dropout=(float, -1.0),
            lstm_layers=(int, 1), lstm_size=(int, 100)),
        EmbeddingParameters)
EvalParameters = Parameters(
        ktags=(int, 5), fallbackprob=(float, 0.0),
        batchsize=(int, 1),
        evalfilename=(str, None), only_disc=(str, "both"), accuracy=(str, "both"), pos_accuracy=(bool, True))
class Supertagger(Model):
    def __init__(self, sequence_tagger: SequenceMultiTagger, grammar: SupertagGrammar):
        super(Supertagger, self).__init__()
        self.sequence_tagger = sequence_tagger
        self.__grammar__ = grammar
        self.__evalparam__ = None
        self.__ktags__ = 1

    def set_eval_param(self, config: EvalParameters):
        self.__grammar__.fallback_prob = config.fallbackprob
        self.__evalparam__ = readparam(config.evalfilename)
        self.__ktags__ = config.ktags

    @property
    def evalparam(self):
        return self.__evalparam__

    @property
    def ktags(self):
        return self.__ktags__

    @classmethod
    def from_corpus(cls, corpus: Corpus, grammar: SupertagGrammar, parameters: ModelParameters):
        """ Construct an instance of the model using
            * supertags and pos tags from `grammar`, and
            * word embeddings (as specified in `parameters`) from `corpus`.
        """
        supertags = Dictionary(add_unk=False)
        for tag in grammar.tags:
            supertags.add_item(tag.str_tag())
        # postags = Dictionary(add_unk=False)
        # for tag in grammar.pos:
        #     postags.add_item(tag)

        rnn_droupout = parameters.lstm_dropout
        if rnn_droupout < 0:
            rnn_droupout = parameters.dropout

        sequence_tagger = SequenceMultiTagger(
            parameters.lstm_size,
            EmbeddingFactory(parameters, corpus),
            [supertags],  #[supertags, postags],
            ["supertag"],  #["supertag", "pos"],
            use_rnn=(parameters.lstm_layers > 0),
            rnn_layers=parameters.lstm_layers,
            dropout=parameters.dropout,
            word_dropout=parameters.word_dropout,
            locked_dropout=parameters.locked_dropout,
            lstm_dropout=rnn_droupout,
            reproject_embeddings=False
        )

        return cls(sequence_tagger, grammar)

    def forward_loss(self, data_points):
        return self.sequence_tagger.forward_loss(data_points)

    def forward(self, data_points):
        return self.sequence_tagger.forward(data_points)

    def predict(self, batch: List[Sentence],
            label_name: str = None,
            return_loss: bool = False,
            embedding_storage_mode: str = "none",
            supertag_storage_mode: str = "both",
            postag_storage_mode: bool = True):
        """ Predicts pos tags and supertags for the given sentences and parses
            them.
            :param label_name: the predicted parse trees are stored in each
                sentence's `label_name` label, the predicted supertags are
                stored in `label_name`-tag of each single token.
            :param return_loss: if true, computes and returns the loss. Gold
                supertags and pos-tags are expected in the `supertag` and `pos`
                labels for each token.
            :param embedding_storage_mode: one of "none", "cpu", "store".
                "none" discards embedding predictions after each batch, "cpu"
                sends the tensors to cpu memory.
            :param supertag_storage_mode: one of "none", "kbest", "best" or
                "both". If "kbest" (or "best"), stores the `self.ktags` best
                (or the best) predicted tags per token. "both" stores the best
                as well as the `self.ktags` per token.
            :param supertag_storage_mode: if set to false, this will not store
                predicted pos tags in each token of the given sentences.
        """
        from flair.training_utils import store_embeddings
        from numpy import argpartition

        if not label_name:
            label_name = "predicted"
        supertag_label_name = f"{label_name}-supertag"
        postag_label_name = f"{label_name}-pos"

        with torch.no_grad():
            scores = dict(self.sequence_tagger.forward(batch))
            tagscores = scores["supertag"].cpu()
            # pos = scores["pos"].argmax(dim=2)
            tags = argpartition(-tagscores, self.ktags, axis=2)[:, :, 0:self.ktags]
            neglogprobs = -tagscores.gather(2, tags).log_softmax(dim=2)
            for sentence, senttags, sentweights in zip(batch, tags, neglogprobs):
                # store tags in tokens
                if supertag_storage_mode in ("both", "kbest", "best"):
                    for token, ktags, kweights, in zip(sentence, senttags, sentweights):
                        ktags = tuple(self.sequence_tagger.type2dict["supertag"].get_item_for_index(t) for t in ktags)
                        kweights = (-kweights).exp()
                        if supertag_storage_mode in ("both", "kbest"):
                            token.add_tags_proba_dist(supertag_label_name, [Label(t, w.item()) for t, w in zip(ktags, kweights)])
                        if supertag_storage_mode in ("both", "best"):
                            best_tag = kweights.argmax()
                            token.add_tag_label(supertag_label_name, Label(ktags[best_tag], kweights[best_tag].item()))
                # if postag_storage_mode:
                #     for token, postag in zip(sentence, sentpos):
                #         strpos = self.sequence_tagger.type2dict["pos"].get_item_for_index(postag)
                #         token.add_tag_label(postag_label_name, Label(strpos))

                # parse sentence and store parse tree in sentence
                sentweights = sentweights[:len(sentence)]
                predicted_tags = (
                    zip(ktags, kweights)
                    for ktags, kweights in zip(senttags[:len(sentence)], sentweights))
                # predicted_pos = [self.__grammar__.pos[tag] for tag in sentpos[:len(sentence)]]
                # parses = self.__grammar__.parse(predicted_pos, predicted_tags, ktags=self.ktags, estimates=sentweights.numpy().min(axis=1))
                parses = self.__grammar__.parse(predicted_tags, ktags=self.ktags, length=len(sentence), estimates=sentweights.numpy().min(axis=1))
                try:
                    parse = str(next(parses))
                except StopIteration:
                    leaves = (f"({p} {i})" for i, p in enumerate(["NP"]*len(sentence)))
                    parse = f"(NOPARSE {' '.join(leaves)})"
                sentence.set_label(label_name, parse)
            store_embeddings(batch, storage_mode=embedding_storage_mode)
            if return_loss:
                return self.sequence_tagger._calculate_loss(scores.items(), batch)

    def evaluate(self,
            sentences: SupertagParseDataset,
            mini_batch_size: int = 32,
            num_workers: int = 1,
            embedding_storage_mode: str = "none",
            out_path = None,
            only_disc: str = "both",
            accuracy: str = "both",
            pos_accuracy: bool = True,
            return_loss: bool = True) -> Tuple[Result, float]:
        """ Predicts supertags, pos tags and parse trees, and reports the
            predictions scores for a set of sentences.
            :param sentences: a ``DataSet`` of sentences. For each sentence
                a gold parse tree is expected as value of the `tree` label, as
                provided by ``SupertagParseDataset``.
            :param only_disc: If set, overrides the setting `DISC_ONLY` in the
                evaluation parameter file ``self.evalparam``, i.e. only evaluates
                discontinuous constituents if True. Pass "both" to report both
                results.
            :param accuracy: either 'none', 'best', 'kbest' or 'both'.
                Determines if the accuracy is computed from the best, or k-best
                predicted tags.
            :param pos_accuracy: if set, reports acc. of predicted pos tags.
            :param return_loss: if set, nll loss wrt. gold tags is reported,
                otherwise the second component in the returned tuple is 0.
            :returns: tuple with evaluation ``Result``, where the main score
                is the f1-score (for all constituents, if only_disc == "both").
        """
        from flair.datasets import DataLoader
        from discodop.tree import ParentedTree, Tree
        from discodop.eval import Evaluator, readparam
        from timeit import default_timer
        from collections import Counter

        if self.__evalparam__ is None:
            raise Exception("Need to specify evaluator parameter file before evaluating")
        if only_disc == "both":
            evaluators = {
                "F1-all":  Evaluator({ **self.evalparam, "DISC_ONLY": False }),
                "F1-disc": Evaluator({ **self.evalparam, "DISC_ONLY": True  })}
        else:
            mode = self.evalparam["DISC_ONLY"] if only_disc == "param" else (only_disc=="true")
            strmode = "F1-disc" if mode else "F1-all"
            evaluators = {
                strmode: Evaluator({ **self.evalparam, "DISC_ONLY": mode })}
        
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # predict supertags and parse trees
        eval_loss = 0
        start_time = default_timer()
        for batch in data_loader:
            loss = self.predict(batch,
                    embedding_storage_mode=embedding_storage_mode,
                    supertag_storage_mode=accuracy,
                    postag_storage_mode=pos_accuracy,
                    label_name='predicted',
                    return_loss=return_loss)
            eval_loss += loss if return_loss else 0
        end_time = default_timer()

        i = 0
        batches = 0
        noparses = 0
        acc_ctr = Counter()
        for batch in data_loader:
            for sentence in batch:
                for token in sentence:
                    if accuracy in ("kbest", "both") and token.get_tag("supertag").value in \
                            (l.value for l in token.get_tags_proba_dist('predicted-supertag')):
                        acc_ctr["kbest"] += 1
                    if accuracy in ("best", "both") and token.get_tag("supertag").value == \
                            token.get_tag('predicted-supertag').value:
                        acc_ctr["best"] += 1
                    if pos_accuracy and token.get_tag("pos").value == token.get_tag("predicted-pos").value:
                        acc_ctr["pos"] += 1
                acc_ctr["all"] += len(sentence)
                sent = [token.text for token in sentence]
                gold = ParentedTree(sentence.get_labels("tree")[0].value)
                parse = Tree(sentence.get_labels("predicted")[0].value)
                parse = ParentedTree.convert(parse)
                if parse.label == "NOPARSE":
                    noparses += 1
                for evaluator in evaluators.values():
                    evaluator.add(i, gold.copy(deep=True), list(sent), parse.copy(deep=True), list(sent))
                i += 1
            batches += 1
        scores = {
            strmode: float_or_zero(evaluator.acc.scores()['lf'])
            for strmode, evaluator in evaluators.items()}
        if accuracy in ("both", "kbest"):
            scores["accuracy-kbest"] = acc_ctr["kbest"] / acc_ctr["all"]
        if accuracy in ("both", "best"):
            scores["accuracy-best"] = acc_ctr["best"] / acc_ctr["all"]
        if pos_accuracy:
            scores["accuracy-pos"] = acc_ctr["pos"] / acc_ctr["all"]
        scores["coverage"] = 1-(noparses/i)
        scores["time"] = end_time - start_time
        return (
            Result(
                scores['F1-all'] if 'F1-all' in scores else scores['F1-disc'],
                "\t".join(f"{mode}" for mode in scores),
                "\t".join(f"{s}" for s in scores.values()),
                '\n\n'.join(evaluator.summary() for evaluator in evaluators.values())),
            eval_loss / batches
        )

    def _get_state_dict(self):
        return {
            "sequence_tagger": self.sequence_tagger._get_state_dict(),
            "grammar": self.__grammar__.__getstate__(),
            "ktags": self.ktags,
            "evalparam": self.evalparam
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        sequence_tagger = SequenceMultiTagger._init_model_with_state_dict(state["sequence_tagger"])
        grammar = SupertagGrammar(state["grammar"]["tags"], state["grammar"]["roots"])
        model = cls(sequence_tagger, grammar)
        model.__ktags__ = state["ktags"]
        model.__evalparam__ = state["evalparam"]
        return model

def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0 # return 0 if f is NaN
    except:
        return 0.0
