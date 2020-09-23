from sys import argv
from tagger import CrfTagger, hyperparam
from flair.datasets.sequence_labeling import ColumnDataset

import logging
logging.getLogger("flair").setLevel(40)

assert len(argv) == 3, f"use {argv[0]} <model file> <data file>"

model = CrfTagger.load(argv[1])
data = ColumnDataset(argv[2], {0: "text", 1: "pos", 2: "preterm", 3: "supertag"})

model.eval()
model.predict(data, mini_batch_size=10, all_tag_prob=False)
for sentence in data:
    for token in sentence:
        print(f"{token.text} {token.get_tag('supertag').value} {token.get_tag('supertag').score}")
    print()