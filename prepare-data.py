from configparser import ConfigParser
from argparse import ArgumentParser
from multiprocessing import Pool

from supertagging.data import corpusparam, SupertagCorpusFile
from gridsearch import Grid

args = ArgumentParser()
args.add_argument("conf", help="configuration file for corpus, e.g. `example.conf`")
args.add_argument("--grid", nargs="+", help="prepare corpora for a grid search")
args.add_argument("-j", help="no. parallel processes for extraction", type=int, default=1)
args = args.parse_args()

cp = ConfigParser()
cp.read(args.conf)
baseconfig = {**cp["Corpus"], **cp["Grammar"]}

if not args.grid:
    with SupertagCorpusFile(corpusparam(**baseconfig)) as corpusfile:
        print("extracted", len(corpusfile.grammar.tags))
        exit()

corpuskeys = corpusparam.keys()
grid = Grid(gridc for gridc in args.grid if gridc.split("=")[0] in corpuskeys)

def extract(gridpoint):
    config = dict(baseconfig)
    for k, v in gridpoint.items():
        config[k] = v
    config = corpusparam(**config)
    try:
        with SupertagCorpusFile(config) as corpusfile:
            return corpusfile.grammar.tags, gridpoint
    except ValueError as e:
        print(e)
        return [], gridpoint

pool = Pool(max(1, args.j))
for tags, gp in pool.imap_unordered(extract, grid):
    print("extracted", len(tags), "for", gp)