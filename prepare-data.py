from supertagging.data import corpusparam, SupertagCorpusFile
from configparser import ConfigParser
from sys import argv

assert len(argv) == 2, (f"use {argv[0]} <data.conf>")

cp = ConfigParser()
cp.read(argv[1])
config = corpusparam(**cp["Corpus"], **cp["Grammar"])

with SupertagCorpusFile(config) as corpusfile:
    print("extracted", len(corpusfile.grammar.tags), "tags")