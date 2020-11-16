# Build

First, build our fork of disco-dop using

    git submodule init && git submodule update  # skip this command, if you did not clone the project, but downloaded a zip-file
    cd disco-dop
    git submodule init && git submodule update  # skip this command, if you did not clone the project, but downloaded a zip-file
    pip install --user -r requirements.txt
    make install
    cd ..

After that, install the remaining requirements for the supertagger

    pip install --user -r requirements.txt

# Usage

A call of

    python prepare-data.py <data.conf>

constructs a lexical grammar, supertags and splits the corpus as specified in ```<data.conf>```.
There are already some configuration files prepared for [Negra](http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/), [Tiger](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger/) and discontinuous Penn Treebank (contact [Kilian Evang](https://kilian.evang.name/)).
However, they require these corpora lying in ```./copora/```.
Depending on the format of the downloaded corpora, you may have to adjust the fields ```inputfmt``` (```export``` vs. ```bracket``` vs. ```tiger```) and ```inputenc``` (```iso-8859-1``` vs. ```utf-8```) in the configuration.
The script writes files to the folder specified in the ```filename``` field of configuration files.

    python training.py <data.conf> <model.conf>

trains a sequence tagger as specified in ```<model.conf>``` using the prepared data.
Existing configurations are ```bilstm-model.conf```, ```bert-model.conf``` and ```supervised-model```.
During training, the script writes checkpoints to a newly created folder named after ```<data.conf>```, ```<model.conf>``` and the current date.
Training is monitored using tensorboard, also there are lists of losses and scores in ```loss.tsv```.

Calling

    python prepare-data.py example.conf
    python training.py example.conf example-model.conf

should word out-of-the-box, as it uses a small (publicly available) sample of the alpino corpus distributed with disco-dop.
Comments on the configuration files can be found in ```example.conf``` and ```example-model.conf```.