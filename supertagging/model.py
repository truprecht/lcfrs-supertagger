from typing import Tuple, Union, List

from flair.data import Dictionary, Label, Sentence
from flair.nn import Model
from flair.training_utils import Result

import torch

from discodop.supertags import SupertagGrammar
from discodop.eval import Evaluator, readparam
from discodop.tree import Tree

from .data import SupertagParseDataset, SupertagParseCorpus
from .parameters import Parameters


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
        evalfilename=(str, None), only_disc=(str, "both"), accuracy=(str, "both"), othertag_accuracy=(bool, True))

def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0 # return 0 if f is NaN
    except:
        return 0.0

def str_or_none(s: str):
    return None if s == "None" else s

def noparse(partial: Tree, postags: list) -> Tree:
    if partial is None:
        return Tree("NOPARSE", [Tree(pt, [i]) for i, pt in enumerate(postags)])
    missing = set(range(len(postags))) - set(partial.leaves())
    if not missing:
        return partial
    return Tree(
        "NOPARSE",
        [partial] + [ Tree(postags[i], [i]) for i in missing ]
    )
