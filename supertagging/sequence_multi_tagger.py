from flair.models.sequence_tagger_model import Dictionary, Sentence, SequenceTagger, TokenEmbeddings, torch, flair
from typing import Iterable, List, Tuple, Union
from torch.nn import LSTM

class SequenceMultiTagger(flair.nn.Model):
    """ A sequence tagger model for multiple tag types per token as
        flair.models.sequence_tagger_model.MultiTagger, but this one uses the
        same architecture for all tags. Only the final linear layer is split
        according to the tag types.
    """
    def __init__(self, hidden_size: int, embeddings: TokenEmbeddings, tag_dictionaries: Iterable[Dictionary], tag_types: Iterable[str], **kwargs):
        super(SequenceMultiTagger, self).__init__()
        newtags = Dictionary(add_unk=False)
        newtypestr = str(tuple(tag_types))
        self.type2slice = {}
        self.type2dict = { t: d for t, d in zip(tag_types, tag_dictionaries) }
        outputsize = 0
        for tagtype, tagdict in zip(tag_types, tag_dictionaries):
            self.type2slice[tagtype] = slice(outputsize, outputsize+len(tagdict))
            outputsize += len(tagdict)
            for tagstr in tagdict.get_items():
                newtag = f"[{tagtype}]-{tagstr}"
                newtags.add_item(newtag)

        for unsupported_param in ("use_crf", "beta", "loss_weights", "lstm_type"):
            assert not unsupported_param in kwargs, f"{unsupported_param} is not supported by SequenceMultiTagger"
        self.inner = SequenceTagger(hidden_size, embeddings, newtags, newtypestr, **kwargs)
        # change fixed dropout of 0.5 in flairs sequence tagger model
        if kwargs.get("use_rnn", False) and kwargs.get("rnn_layers", 1) > 1:
            self.inner.rnn = LSTM(
                embeddings.embedding_length,
                hidden_size,
                num_layers=kwargs["rnn_layers"],
                dropout=kwargs.get("dropout", 0.0),
                bidirectional=True,
                batch_first=True)
        self.to(flair.device)

    def predict(*args, **kwargs):
        raise NotImplementedError()

    def evaluate(*args, **kwargs):
        raise NotImplementedError()
    
    def _get_state_dict(self):
        return {
            "inner": self.inner._get_state_dict(),
            "type2dict": self.type2dict,
            "type2slice": self.type2slice,
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls.__new__(cls)
        super(SequenceMultiTagger, model).__init__()
        model.type2dict = state["type2dict"]
        model.type2slice = state["type2slice"]

        inner = state["inner"]
        model.inner = SequenceTagger(
            hidden_size=inner["hidden_size"],
            embeddings=inner["embeddings"],
            tag_dictionary=inner["tag_dictionary"],
            tag_type=inner["tag_type"],
            use_crf=inner["use_crf"],
            use_rnn=inner["use_rnn"],
            rnn_layers=inner["rnn_layers"],
            dropout=inner.get("use_dropout", 0.0),
            word_dropout=inner.get("use_word_dropout", 0.0),
            locked_dropout=inner.get("use_locked_dropout", 0.0),
            train_initial_hidden_state=inner.get("train_initial_hidden_state", False),
            rnn_type=inner.get("rnn_type", "LSTM"),
            beta=inner.get("beta", 1.0),
            loss_weights=inner.get("weights", None),
            reproject_embeddings=inner.get("reproject_embeddings", True),)
        if model.inner.use_rnn and model.inner.rnn_layers > 1:
            model.inner.rnn = LSTM(
                model.inner.embeddings.embedding_length,
                model.inner.hidden_size,
                num_layers=model.inner.rnn_layers,
                dropout=model.inner.use_dropout,
                bidirectional=True,
                batch_first=True)
        model.inner.load_state_dict(inner["state_dict"])
        model.to(flair.device)
        return model

    def _calculate_loss(self,
            feats_per_type: Iterable[Tuple[str, torch.tensor]],
            batch: Union[List[Sentence], Sentence]) -> torch.tensor:
        loss = torch.tensor(0, dtype=float, device=flair.device)
        for tagtype, feats in feats_per_type:
            for sentence_feats, sentence in zip(feats, batch):
                strtags = [token.get_tag(tagtype).value for token in sentence]
                idxtags = torch.tensor(self.type2dict[tagtype].get_idx_for_items(strtags), device=flair.device)
                sentence_feats = sentence_feats[:len(sentence)]
                loss += torch.nn.functional.cross_entropy(sentence_feats, idxtags)
        return loss / len(batch)

    def forward(self, batch: List[Sentence]) -> Iterable[Tuple[str, torch.tensor]]:
        feats = self.inner.forward(batch)
        for tagtype, loc in self.type2slice.items():
            yield (tagtype, feats[:, :, loc])

    def forward_loss(self, batch: List[Sentence]) -> torch.tensor:
        return self._calculate_loss(self.forward(batch), batch)
