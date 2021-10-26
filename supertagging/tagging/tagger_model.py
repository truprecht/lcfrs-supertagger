import torch
import flair

from typing import Tuple, Iterable

from discodop.eval import readparam
from discodop.supertags import SupertagGrammar

from ..data import SupertagParseDataset, SupertagParseCorpus
from ..model import float_or_zero, parse_or_none
from . import decoder
from .encoder import BilstmEncoder
from ..parameters import Parameters
from .embeddings import EmbeddingBuilder, EmbeddingParameters


EvalParameters = Parameters(
        ktags=(int, 10), fallbackprob=(float, 0.0), batchsize=(int, 32),
        evalfilename=(str, "disco-dop/proper.prm"), only_disc=(str, "both"), accuracy=(str, "none"), othertag_accuracy=(bool, False))
DecoderModelParameters = Parameters.merge(
    Parameters(
        dropout=(float, 0.0), word_dropout=(float, 0.0), locked_dropout=(float, 0.0), lstm_dropout=(float, 0.0),
        lstm_layers=(int, 1), lstm_size=(int, 100),
        decoder_hidden_dim=(int, 100), decoder_embedding_dim=(int, 100), sample_gold_tags=(float, 0.0), decodertype=(str, "FfDecoder")),
    EmbeddingParameters)
class DecoderModel(flair.nn.Model):
    @classmethod
    def from_corpus(cls, corpus: SupertagParseCorpus, grammar: SupertagGrammar, parameters: DecoderModelParameters, additional_tags: Iterable[str]):
        """ Construct an instance of the model using
            * supertags and pos tags from `grammar`, and
            * word embeddings (as specified in `parameters`) from `corpus`.
        """
        tag_dicts = { k: flair.data.Dictionary(add_unk=True) for k in ("supertag",) + tuple(additional_tags) }
        for str_tag in grammar.str_tags:
            tag_dicts["supertag"].add_item(str_tag)
        for sentence in corpus.train:
            for token in sentence:
                for k in tag_dicts:
                    tag_dicts[k].add_item(token.get_tag(k).value)

        kwargs = [ "decoder_embedding_dim", "decoder_hidden_dim", "dropout", "word_dropout", "locked_dropout", "lstm_dropout", "sample_gold_tags", "decodertype" ]
        kwargs = { kw: parameters.__getattribute__(kw) for kw in kwargs }

        return DecoderModel(EmbeddingBuilder(parameters, corpus), tag_dicts, grammar,
            encoder_layers=parameters.lstm_layers, encoder_hidden_dim=parameters.lstm_size,
            **kwargs)

    def __init__(self, embedding_builder, tag_dicts, grammar, encoder_layers=1, encoder_hidden_dim=100, decoder_hidden_dim=100, decoder_embedding_dim=100, dropout=0.0, word_dropout=0.0, locked_dropout=0.0, lstm_dropout=0.0, sample_gold_tags = 0.0, decodertype: str = "FfDecoder"):
        super(DecoderModel, self).__init__()

        self.sample_gold_tags = sample_gold_tags

        self.embedding_builder = embedding_builder
        self.encoder_layers = encoder_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decodertype = decodertype
        decodertype = getattr(decoder, self.decodertype)

        self.embedding = self.embedding_builder.produce()

        self.dropout = dropout
        self.word_dropout = word_dropout
        self.locked_dropout = locked_dropout
        self.lstm_dropout = lstm_dropout
        
        dropout_layers = []
        if dropout > 0.0:
            dropout_layers.append(torch.nn.Dropout(dropout))
        if word_dropout > 0.0:
            dropout_layers.append(flair.nn.WordDropout(word_dropout))
        if locked_dropout > 0.0:
            dropout_layers.append(flair.nn.LockedDropout(locked_dropout))
        self.dropout_layers = torch.nn.Sequential(*dropout_layers)

        self.dictionaries = tag_dicts
        inputlen = self.embedding.embedding_length

        self.encoder = BilstmEncoder(inputlen, encoder_hidden_dim, layers=encoder_layers, dropout=lstm_dropout)
        self.decoder = decodertype(self.encoder.output_dim, len(tag_dicts["supertag"]), decoder_embedding_dim, decoder_hidden_dim)
        self.othertaggers = torch.nn.ModuleDict({
            name: torch.nn.Linear(self.encoder.output_dim, len(dictionary))
            for name, dictionary in self.dictionaries.items()
            if name != "supertag"
        })

        self.__grammar__ = grammar
        self.__evalparam__ = None
        self.__ktags__ = 1

        self.to(flair.device)

    def set_eval_param(self, config: EvalParameters):
        self.__evalparam__ = readparam(config.evalfilename)
        self.__ktags__ = config.ktags

    @property
    def evalparam(self):
        return self.__evalparam__

    @property
    def ktags(self):
        return self.__ktags__


    def label_type(self):
        return "supertag"


    def _batch_to_embeddings(self, batch):
        if not type(batch) is list:
            batch = [batch]
        self.embedding.embed(batch)
        embedding_name = self.embedding.get_names()
        input = torch.nn.utils.rnn.pad_sequence([
            torch.stack([ word.get_embedding(embedding_name) for word in sentence ])
            for sentence in batch]).to(flair.device)
        return input


    def _batch_to_gold(self, batch, batch_first=False, padding_value=-100):
        if not type(batch) is list:
            batch = [batch]

        for tagname, tagdict in self.dictionaries.items():
            mat = torch.nn.utils.rnn.pad_sequence([
                torch.tensor([ tagdict.get_idx_for_item(word.get_tag(tagname).value) for word in sentence ])
                for sentence in batch], padding_value=padding_value).to(flair.device)
            if batch_first:
                mat = mat.transpose(0,1)
            yield tagname, mat


    def _calculate_loss(self, feats, batch, batch_first=False, mean=True):
        loss = torch.tensor(0.0, device=flair.device)
        feats = dict(feats)
        for tagname, golds in self._batch_to_gold(batch, batch_first):
            loss += torch.nn.functional.cross_entropy(feats[tagname].flatten(end_dim=1), golds.flatten(end_dim=1), reduction="sum", ignore_index=-100)
        n_predictions = sum(len(sentence) for sentence in batch)
        if mean:
            return loss / n_predictions
        else:
            return loss, n_predictions


    def forward_loss(self, batch):
        if self.sample_gold_tags > 0.0:
            stags = next(tensor for name, tensor in self._batch_to_gold(batch, padding_value=0) if name == "supertag")
        else:
            stags = None
        feats = self.forward(batch, gold_outputs=stags)
        return self._calculate_loss(feats, batch, mean=True)


    def forward(self, batch, batch_first=False, gold_outputs=None):
        inputfeats = self._batch_to_embeddings(batch)
        inputfeats = self.dropout_layers(inputfeats)
        inputfeats = self.encoder(inputfeats)
        inputfeats = self.dropout_layers(inputfeats)
        stagfeats = self.decoder(inputfeats, gold_outputs=gold_outputs, gold_sampling_prob=self.sample_gold_tags if not gold_outputs is None else 0.0)
        yield "supertag", stagfeats if not batch_first else stagfeats.transpose(0,1)
        for name, decoder in self.othertaggers.items():
            feats = decoder(inputfeats)
            yield name, feats if not batch_first else feats.transpose(0,1)


    def predict(self, batch,
            label_name: str = None,
            return_loss: bool = False,
            embedding_storage_mode: str = "none",
            supertag_storage_mode: str = "both",
            othertag_storage_mode: bool = True):
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
        get_label_name = lambda x: f"{label_name}-{x}"

        with torch.no_grad():
            scores = dict(self.forward(batch, batch_first=True))
            tagscores = scores["supertag"].cpu()
            tags = argpartition(-tagscores, self.ktags, axis=2)[:, :, 0:self.ktags]
            neglogprobs = -tagscores.gather(2, tags).log_softmax(dim=2)
            othertags = (
                {
                    name: [
                        parse_or_none(self.dictionaries[name].get_item_for_index(idx), int if name == "transport" else str)
                        for idx in tscores[sentidx, :len(sent)].argmax(dim=-1)]
                    for name, tscores in scores.items() if name != "supertag"
                } for sentidx, sent in enumerate(batch) )

            for sentence, senttags, sentweights, othertag in zip(batch, tags, neglogprobs, othertags):
                # store tags in tokens
                if supertag_storage_mode in ("both", "kbest", "best"):
                    for token, ktags, kweights in zip(sentence, senttags, sentweights):
                        ktags = tuple(self.dictionaries["supertag"].get_item_for_index(t) for t in ktags)
                        kweights = (-kweights).exp()
                        if supertag_storage_mode in ("both", "kbest"):
                            token.add_tags_proba_dist(get_label_name("supertag"), [flair.data.Label(t, w.item()) for t, w in zip(ktags, kweights)])
                        if supertag_storage_mode in ("both", "best"):
                            best_tag = kweights.argmax()
                            token.add_tag_label(get_label_name("supertag"), flair.data.Label(ktags[best_tag], kweights[best_tag].item()))
                if othertag_storage_mode:
                    for tagt in self.dictionaries:
                        if tagt == "supertag": continue
                        for token, tagstr in zip(sentence, othertag[tagt]):
                            token.add_tag_label(get_label_name(tagt), flair.data.Label(str(tagstr)))

                # parse sentence and store parse tree in sentence
                sentweights = sentweights[:len(sentence)]
                predicted_tags = (
                    ((tag-1, weight) for tag, weight in zip(ktags, kweights) if tag != 0)
                    for ktags, kweights in zip(senttags[:len(sentence)], sentweights))

                parses = self.__grammar__.parse(predicted_tags, **othertag, ktags=self.ktags, length=len(sentence), estimates=sentweights.numpy().min(axis=1))
                try:
                    treeparse = next(parses)
                    if not "pos" in self.dictionaries:
                        postags = list(p for _, p in sorted(treeparse.pos()))
                    parse = str(treeparse)
                except StopIteration:
                    if "pos" in othertag:
                        leaves = (f"({p} {i})" for i, p in enumerate(othertag["pos"]))
                    else:
                        leaves = (f"({p} {i})" for i, p in enumerate(["NOPARSE"]*len(sentence)))
                        postags = ["<UNK>"]*len(sentence)
                    parse = f"(NOPARSE {' '.join(leaves)})"
                sentence.set_label(label_name, parse)
                
                if not "pos" in self.dictionaries and othertag_storage_mode:
                    for token, pos in zip(sentence, postags):
                        token.add_tag_label(get_label_name("pos"), flair.data.Label(pos))

            store_embeddings(batch, storage_mode=embedding_storage_mode)
            if return_loss:
                return self._calculate_loss(scores.items(), batch, batch_first=True, mean=False)

    def evaluate(self,
            sentences: SupertagParseDataset,
            gold_label_type: str = "supertag",
            mini_batch_size: int = 32,
            num_workers: int = 1,
            embedding_storage_mode: str = "none",
            out_path = None,
            only_disc: str = "both",
            accuracy: str = "both",
            othertag_accuracy: bool = True,
            main_evaluation_metric = (),
            gold_label_dictionary = None,
            return_loss: bool = True) -> Tuple[flair.training_utils.Result, float]:
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
        from discodop.eval import Evaluator
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
        n_predictions = 0
        start_time = default_timer()

        for batch in data_loader:
            loss = self.predict(batch,
                    embedding_storage_mode=embedding_storage_mode,
                    supertag_storage_mode=accuracy,
                    othertag_storage_mode=othertag_accuracy,
                    label_name='predicted',
                    return_loss=return_loss)
            if return_loss:
                eval_loss += loss[0]
                n_predictions += loss[1]
        end_time = default_timer()
        if return_loss:
            eval_loss /= n_predictions

        othertags = (set(self.dictionaries) - {"supertag"}) | { "pos" } if othertag_accuracy else {}

        i = 0
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
                    for tagt in othertags:
                        acc_ctr[tagt] += int(token.get_tag(tagt).value == token.get_tag(f"predicted-{tagt}").value)
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
        scores = {
            strmode: float_or_zero(evaluator.acc.scores()['lf'])
            for strmode, evaluator in evaluators.items()}
        if accuracy in ("both", "kbest"):
            scores["accuracy-kbest"] = acc_ctr["kbest"] / acc_ctr["all"]
        if accuracy in ("both", "best"):
            scores["accuracy-best"] = acc_ctr["best"] / acc_ctr["all"]
        for tagt in othertags:
            scores[f"accuracy-{tagt}"] = acc_ctr[tagt] / acc_ctr["all"]
        scores["coverage"] = 1-(noparses/i)
        scores["time"] = end_time - start_time

        result_args = dict(
            main_score=scores['F1-all'] if 'F1-all' in scores else scores['F1-disc'],
            log_header="\t".join(f"{mode}" for mode in scores),
            log_line="\t".join(f"{s}" for s in scores.values()),
            detailed_results='\n\n'.join(evaluator.summary() for evaluator in evaluators.values()))
        
        if flair.__version__ >= "0.9":
            return flair.training_utils.Result(**result_args, loss=eval_loss, classification_report=None)
        return flair.training_utils.Result(**result_args), eval_loss

    def _get_state_dict(self):
        return {
            "state_dict": self.state_dict(),
            "embedding_builder": self.embedding_builder,
            "decoder_embedding_dim": self.decoder_embedding_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "encoder_layers": self.encoder_layers,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "dropout": self.dropout,
            "locked_dropout": self.locked_dropout,
            "word_dropout": self.word_dropout,
            "lstm_dropout": self.lstm_dropout,
            "tags": self.dictionaries,
            "ktags": self.ktags,
            "evalparam": self.evalparam,
            "sample_gold_tags": self.sample_gold_tags,
            "decodertype": self.decodertype,
            "grammar": self.__grammar__.__getstate__()
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        args = state["embedding_builder"], state["tags"], SupertagGrammar(state["grammar"]["tags"], state["grammar"]["roots"])
        kwargs = {
            kw: state[kw]
            for kw in ("encoder_hidden_dim", "encoder_layers", "decoder_hidden_dim",
                "decoder_embedding_dim", "dropout", "word_dropout", "locked_dropout",
                "lstm_dropout", "sample_gold_tags", "decodertype") }
        model = cls(*args, **kwargs)
        model.__ktags__ = state["ktags"]
        model.__evalparam__ = state["evalparam"]
        model.load_state_dict(state["state_dict"])
        return model
