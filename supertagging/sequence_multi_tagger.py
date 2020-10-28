from flair.models.sequence_tagger_model import *
from typing import Iterable, Tuple

class SequenceMultiTagger(SequenceTagger):
    def __init__(self, hidden_size, embeddings, tag_dictionaries, tag_types, **kwargs):
        # exclude args that don't make sense anymore
        # TODO: the first two are just due to the way we instantiate this class
        assert "hidden_size" not in kwargs
        assert "embeddings" not in kwargs
        assert "tag_dictionary" not in kwargs
        assert "tag_types" not in kwargs
        assert "loss_weights" not in kwargs
        assert "use_crf" not in kwargs

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

        super(SequenceMultiTagger, self).__init__(hidden_size, embeddings, newtags, newtypestr, use_crf=False, **kwargs)

        self.to(flair.device)

    def predict(*args, **kwargs):
        raise NotImplementedError()

    def evaluate(*args, **kwargs):
        raise NotImplementedError()
    
    def _get_state_dict(self):
        raise NotImplementedError()

    @classmethod
    def _init_model_with_state_dict(cls, state):
        raise NotImplementedError()

    def _calculate_loss(self, feats_per_type: Iterable[Tuple[str, torch.tensor]], batch: Union[List[Sentence], Sentence]) -> torch.tensor:
        loss = torch.tensor(0, dtype=float)
        for tagtype, feats in feats_per_type:
            for sentence_feats, sentence in zip(feats, batch):
                strtags = [token.get_tag(tagtype).value for token in sentence]
                idxtags = torch.tensor(self.type2dict[tagtype].get_idx_for_items(strtags), device=flair.device)
                sentence_feats = sentence_feats[:len(sentence)]
                loss += torch.nn.functional.cross_entropy(sentence_feats, idxtags)
        return loss / len(batch)

    def forward(self, batch: List[Sentence]) -> Iterable[Tuple[str, torch.tensor]]:
        feats = super(SequenceMultiTagger, self).forward(batch)
        for tagtype, loc in self.type2slice.items():
            yield (tagtype, feats[:, :, loc])
