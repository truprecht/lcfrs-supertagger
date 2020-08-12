import torch
from training import read_config, load_data
from tagger import tagger
from dataset import SupertagDataset, embedding_factory

from discodop.lexcorpus import SupertagCorpus, to_parse_tree
from discodop.lexgrammar import SupertagGrammar
from discodop.treebanktransforms import removefanoutmarkers

def first_or_noparse(derivations, sentence):
    try:
        deriv = next(derivations)
        deriv = to_parse_tree(deriv)
        return removefanoutmarkers(deriv)
    except StopIteration:
        return f"(NOPARSE {' '.join(sentence)})"
    except Exception as e:
        print(e)

if __name__ == "__main__":
    config = read_config()
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (_, _, val_data), embedding, dims, _ = load_data(config)
    grammar = SupertagGrammar(SupertagCorpus.read(open(config["Data"]["corpus"], "rb")))

    model = tagger(dims, embedding.vectors, tagger.Hyperparameters.from_dict(config["Training"]))
    state = torch.load(config["Training"]["save_file"], map_location=torch_device)
    model.load_state_dict(state["model"])
    model.eval()

    k = int(config["Val"]["top_tags"])

    with torch.no_grad():
        gold = []
        prediction = []
        for sample in val_data:
            words, trees, wordembeddings, pos, prets, stags, lens = sample
            (wordembeddings, pos, prets, stags, lens) = (t.to(torch_device) for t in (wordembeddings, pos, prets, stags, lens))
            (pret_scores, stag_scores) = model((wordembeddings, pos, lens))
            preterminals, supertags, weights = tagger.n_best_tags((pret_scores, stag_scores), k)
            for batch_idx, sequence_len in enumerate(lens):
                sequence_preterminals = preterminals[0:sequence_len, batch_idx].numpy()
                sequence_supertags = supertags[0:sequence_len, batch_idx]
                sequence_weights = supertags[0:sequence_len, batch_idx]
                sequence_pos = pos[0:sequence_len, batch_idx].numpy()
                derivs = grammar.parse(words[batch_idx], sequence_pos, sequence_preterminals, sequence_supertags, sequence_weights, 1)
                # prediction.append(first_or_noparse(derivs, words[batch_idx]))
                # gold.append(trees[batch_idx])
                print(first_or_noparse(derivs, words[batch_idx]))
        # evaluate
