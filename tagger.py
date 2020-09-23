from parameters import Parameters

from flair.models import SequenceTagger
from flair.embeddings import CharacterEmbeddings, StackedEmbeddings, WordEmbeddings, OneHotEmbeddings
from flair.nn import Model

from flair.datasets.sequence_labeling import Corpus
from flair.data import Dictionary

hyperparam = Parameters(
        pos_embedding_dim=(int, 10), dropout=(float, 0.1),
        char_embedding_dim=(int, 30), wordembedding=(str, "glove"),
        lstm_layers=(int, 1), lstm_size=(int, 100))
class CrfTagger(Model):
    def __init__(self, embeddings, lstm_size: int, lstm_layers: int, tags: Dictionary, dropout: float):
        super(CrfTagger, self).__init__()
        self.supertags = SequenceTagger(
            lstm_size, embeddings, tags, "supertag",
            rnn_layers=lstm_layers, dropout=dropout
        )

    @classmethod
    def from_corpus(cls, corpus: Corpus, parameters: hyperparam):
        emb = StackedEmbeddings([
            CharacterEmbeddings(char_embedding_dim=parameters.char_embedding_dim),
            WordEmbeddings(parameters.wordembedding),
            OneHotEmbeddings(corpus, field="pos", embedding_length=parameters.pos_embedding_dim)])
        return cls(emb, parameters.lstm_size, parameters.lstm_layers,
            corpus.make_tag_dictionary(tag_type="supertag"), parameters.dropout)

    def forward_loss(self, data_points):
        return self.supertags.forward_loss(data_points)

    def evaluate(self, sentences, **kvargs):
        return self.supertags.evaluate(sentences, **kvargs)

    def _get_state_dict(self):
        return {
            "state": self.state_dict(),
            "embeddings": self.supertags.embeddings,
            "lstm_size": self.supertags.hidden_size,
            "lstm_layers": self.supertags.rnn_layers,
            "tags": self.supertags.tag_dictionary,
            "dropout": self.supertags.dropout.p
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        return cls(state["embeddings"], state["lstm_size"], state["lstm_layers"], state["tags"], state["dropout"])

    def predict(self, sentences, **kvargs):
        return self.supertags.predict(sentences, **kvargs)