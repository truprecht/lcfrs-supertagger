from ..parameters import Parameters

import torch
import flair
from flair.data import Corpus, Dictionary
from flair.embeddings import TokenEmbeddings, FlairEmbeddings, StackedEmbeddings, \
        WordEmbeddings, OneHotEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings

from abc import ABC, abstractmethod
from collections import Counter


class SerializableOneHotEmbeddings(OneHotEmbeddings):
    @classmethod
    def build_dictionary(self, corpus: Corpus, field: str, min_freq: int = 1):
        tokens = (
            token.text if field == "text" else token.get_tag(field).value
            for sentence in corpus.train
            for token in sentence
        )
        count = Counter()
        vocab = Dictionary(add_unk=True)
        for token in tokens:
            count[token] += 1
            if count[token] >= min_freq:
                vocab.add_item(token)
        return vocab

    def __init__(self, vocab: Dictionary, field: str, length: int):
        empty_corpus = flair.data.Corpus([])
        super().__init__(empty_corpus, field=field, embedding_length=length)
        self.vocab_dictionary = vocab
        self.embedding_layer = torch.nn.Embedding(len(self.vocab_dictionary), self.embedding_length)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)
        self.to(flair.device)


EmbeddingParameters = Parameters(
    embedding=(str, "word char"), tune_embedding=(bool, False), language=(str, ""),
    pos_embedding_dim=(int, 20),
    word_embedding_dim=(int, 300), word_minfreq=(int, 1),
    char_embedding_dim=(int, 64), char_bilstm_dim=(int, 100))


class TokenEmbeddingBuilder(ABC):
    @abstractmethod
    def __init__(self, name: str, corpus: Corpus, parameters: EmbeddingParameters):
        raise NotImplementedError()

    @abstractmethod
    def produce(self) -> TokenEmbeddings:
        raise NotImplementedError()


class PretrainedBuilder(TokenEmbeddingBuilder):
    @classmethod
    def transformer_str(cls, modelstr: str, language_code: str):
        if modelstr != "bert-base":
            return modelstr
        # translate two letter language code into bert model
        return {
            "de": "bert-base-german-cased",
            "en": "bert-base-cased",
            "nl": "wietsedv/bert-base-dutch-cased"
        }[language_code]

    def __init__(self, name: str, corpus: Corpus, parameters: EmbeddingParameters):
        if any((spec in name) for spec in ("bert", "gpt", "xlnet")):
            self.embedding_t = TransformerWordEmbeddings
            self.model_str = self.__class__.transformer_str(name, parameters.language)
        elif name == "flair":
            self.embedding_t = FlairEmbeddings
            self.model_str = parameters.language
        elif name == "fasttext":
            self.embedding_t = WordEmbeddings
            self.model_str = parameters.language
        else:
            raise NotImplementedError(f"Cound not recognize embedding {name}")
        self.tune = parameters.tune_embedding

    def produce(self) -> TokenEmbeddings:
        if self.embedding_t is TransformerWordEmbeddings:
            return TransformerWordEmbeddings(model=self.model_str, fine_tune=self.tune)
        if self.embedding_t is FlairEmbeddings:
            return StackedEmbeddings([
                FlairEmbeddings(f"{self.model_str}-forward", fine_tune=self.tune),
                FlairEmbeddings(f"{self.model_str}-backward", fine_tune=self.tune)])
        if self.embedding_t is WordEmbeddings:
            return WordEmbeddings(self.model_str)


class CharacterEmbeddingBuilder(TokenEmbeddingBuilder):
    def __init__(self, name: str, corpus: Corpus, parameters: EmbeddingParameters):
        self.embedding_dim = parameters.char_embedding_dim
        self.hidden_size = parameters.char_bilstm_dim

    def produce(self) -> TokenEmbeddings:
        return CharacterEmbeddings(
            char_embedding_dim=self.embedding_dim,
            hidden_size_char=self.hidden_size)


class OneHotEmbeddingBuilder(TokenEmbeddingBuilder):
    def __init__(self, name: str, corpus: Corpus, parameters: EmbeddingParameters):
        self.field = "text" if name == "word" else name
        self.min_freq = parameters.word_minfreq if name == "word" else 1
        self.vocab = SerializableOneHotEmbeddings.build_dictionary(corpus, self.field, self.min_freq)
        self.length = parameters.__getattribute__(f"{name}_embedding_dim")

    def produce(self) -> TokenEmbeddings:
        return SerializableOneHotEmbeddings(self.vocab, self.field, self.length)


class EmbeddingBuilder:
    NAME_TO_CLASS = {
        "word": OneHotEmbeddingBuilder, "pos": OneHotEmbeddingBuilder,
        "char": CharacterEmbeddingBuilder
    }

    def __init__(self, parameters: EmbeddingParameters, corpus: Corpus):
        self.stack = []
        for name in parameters.embedding.split():
            builder = self.__class__.NAME_TO_CLASS.get(name, PretrainedBuilder)
            self.stack.append(builder(name, corpus, parameters))

    def produce(self) -> TokenEmbeddings:
        return StackedEmbeddings([builder.produce() for builder in self.stack])