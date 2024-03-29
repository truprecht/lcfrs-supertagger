# LCFRS Supertag Parser

This projects implements an extraction of lexical lcfrs rules from corpora of constituent trees, training of lexical rule prediction (supertagging) and parsing using lexical rules.
The parts concerning grammars (extraction, parsing) were implemented as an extension of [disco-dop](https://github.com/andreasvc/disco-dop), the prediction was implemented using the [flair framework](https://github.com/flairNLP/flair) for neural nlp.

## Build

The project was developed and tested using python 3.8.
We strongly recommend using a conda (or virtualenv) environment when running it:

    conda create -n lcfrs-supertagger python=3.8 gcc_linux-64=9.3 gxx_linux-64=9.3 && conda activate lcfrs-supertagger
    # or virtualenv: virtualenv venv && ./venv/bin/activate

Build and install all dependencies, including our fork of disco-dop:

    # the following should be skipped if you did not use git to obain these files
    git submodule update --init --recursive
    pip install cython `cat requirements.txt` && pip install ./disco-dop/

## Usage

A call of

    python prepare-data.py <data.conf>

constructs a lexical grammar, supertags and splits the corpus as specified in ```<data.conf>```.
There are already some configuration files prepared for [Negra](http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/), [Tiger](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger/) and discontinuous Penn Treebank (contact [Kilian Evang](https://kilian.evang.name/)) in ```./config/corpus/```.
However, they require these corpora lying in ```./copora/```.
Depending on the format of the downloaded corpora, you may have to adjust the fields ```inputfmt``` (```export``` vs. ```bracket``` vs. ```tiger```) and ```inputenc``` (```iso-8859-1``` vs. ```utf-8```) in the configuration.
The script writes files to the folder specified in the ```filename``` field of configuration files.

    python training.py <data.conf> <model.conf>

trains a sequence tagger as specified in ```<model.conf>``` using the prepared data.
Existing configurations are ```bilstm-model.conf```, ```bert-model.conf```, ```supervised-small.conf``` and  ```supervised-large.conf``` in ```./config/model/```.
During training, the script writes checkpoints to a newly created folder named after ```<data.conf>```, ```<model.conf>``` and the current date.
Training is monitored using tensorboard, also there are lists of losses and scores in ```loss.tsv```.

When the training finishes, the model is automatically evaluated on the test set and parsing scores are reported.
You may repeat this evaluation (eg. with changed evaluation parameters or changed data) by calling

    python evaluate.py <data.conf> <model>

where ```<model>``` is a model file from the checkpoint folder (eg. ```trained-corpus-model-date/best-model.pt```).

Calling

    python prepare-data.py config/corpus/example.conf
    python training.py config/model/example.conf config/corpus/example.conf

should work out-of-the-box, as it uses a small (publicly available) sample of the alpino corpus distributed with disco-dop.
Comments on the configuration files can be found in ```config/model/example.conf``` and ```config/corpus/example.conf```.

### Tiger corpus

This treebank needs a speacial treatment, because some nodes in the treebank are linked to multiple parents.
The issue is solved by removing overfluous links (cf. https://github.com/mcoavoux/multilingual_disco_data/blob/master/generate_tiger_data.sh):

    sed -e "3097937d;3097954d;3097986d;3376993d;3376994d;3377000d;3377001d;3377002d;3377008d;3377048d;3377055d" tiger.xml > tiger-fixed.xml