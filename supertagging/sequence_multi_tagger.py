from flair.models.sequence_tagger_model import Dictionary, Sentence, SequenceTagger, torch, flair
from typing import Iterable, List, Tuple, Union

class SequenceMultiTagger(flair.nn.Model):
    def __init__(self, hidden_size, embeddings, tag_dictionaries, tag_types, **kwargs):
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

        self.inner = SequenceTagger(hidden_size, embeddings, newtags, newtypestr, **kwargs)
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
        model.inner = SequenceTagger._init_model_with_state_dict(state["inner"])
        model.type2dict = state["type2dict"]
        model.type2slice = state["type2slice"]
        model.to(flair.device)
        return model

    def _calculate_loss(self, feats_per_type: Iterable[Tuple[str, torch.tensor]], batch: Union[List[Sentence], Sentence]) -> torch.tensor:
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
