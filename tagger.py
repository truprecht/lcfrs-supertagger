from parameters import Parameters

from flair.models import SequenceTagger
from flair.embeddings import CharacterEmbeddings, StackedEmbeddings, WordEmbeddings, OneHotEmbeddings
from flair.nn import Model

from flair.datasets.sequence_labeling import Corpus
from flair.data import Dictionary

from discodop.lexgrammar import SupertagGrammar

hyperparam = Parameters(
        pos_embedding_dim=(int, 10), dropout=(float, 0.1),
        char_embedding_dim=(int, 30), wordembedding=(str, "glove"),
        lstm_layers=(int, 1), lstm_size=(int, 100))
class Supertagger(Model):
    def __init__(self, embeddings, grammar, lstm_size: int, lstm_layers: int, tags: Dictionary, dropout: float):
        super(Supertagger, self).__init__()
        self.supertags = SequenceTagger(
            lstm_size, embeddings, tags, "supertag",
            rnn_layers=lstm_layers, dropout=dropout
        )
        self.grammar = grammar

    @classmethod
    def from_corpus(cls, corpus: Corpus, grammar: SupertagGrammar, parameters: hyperparam):
        emb = StackedEmbeddings([
            # CharacterEmbeddings(char_embedding_dim=parameters.char_embedding_dim),
            WordEmbeddings(parameters.wordembedding),
            OneHotEmbeddings(corpus, field="pos", embedding_length=parameters.pos_embedding_dim)])
        # avoid unk token in dictionary
        tags = corpus.make_label_dictionary("supertag")
        return cls(emb, grammar, parameters.lstm_size, parameters.lstm_layers,
            tags, parameters.dropout)

    def forward_loss(self, data_points):
        return self.supertags.forward_loss(data_points)

    def evaluate(self, sentences, mini_batch_size=32, num_workers=1, embedding_storage_mode="none", out_path=None):
        from flair.data import Dataset
        from flair.datasets import SentenceDataset, DataLoader
        from flair.training_utils import Result
        from discodop.lexcorpus import to_parse_tree
        from discodop.tree import ParentedTree
        from discodop.treetransforms import unbinarize, removefanoutmarkers
        from discodop.eval import Evaluator, readparam
        
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # else, use scikit-learn to evaluate
        y_true = []
        y_pred = []
        eval_loss = 0
        i = 0
        batches = 0
        evaluator = Evaluator(readparam("proper.prm"))
        for batch in data_loader:
            # predict for batch
            loss = self.predict(batch,
                                embedding_storage_mode=embedding_storage_mode,
                                mini_batch_size=mini_batch_size,
                                label_name='predicted',
                                return_loss=True)
            eval_loss += loss

            for sentence in batch:
                predicted_tags = []
                for token in sentence:
                    # add gold tag
                    gold_tag = token.get_tag("supertag").value
                    y_true.append(gold_tag)

                    # add predicted tag
                    predicted_tag = token.get_tag('predicted').value
                    predicted_score = token.get_tag('predicted').score
                    predicted_tags.append([(predicted_tag, predicted_score)])
                    y_pred.append(predicted_tag)
                sent = [token.text for token in sentence]
                pos = [token.get_tag("pos").value for token in sentence]
                parses = self.grammar.parse(sent, pos, pos, predicted_tags)
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
        print(sum(1 for p, g in zip(y_pred, y_true) if p == g), sum(1 for p, g in zip(y_pred, y_true) if p != g))
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
            "grammar": self.grammar.todict()
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls(
            state["embeddings"],
            SupertagGrammar.fromdict(state["grammar"]),
            state["lstm_size"], state["lstm_layers"], state["tags"], state["dropout"])
        model.load_state_dict(state["state"])
        return model

    def predict(self, sentences, **kvargs):
        return self.supertags.predict(sentences, **kvargs)


def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0
    except:
        return 0.0
