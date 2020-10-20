# Build

First, build our fork of disco-dop using

    git submodule update
    cd disco-dop
    git submodule update
    pip install --user -r requirements.txt
    make install
    cd ..

After that, install the remaining requirements for the supertagger

    pip install --user -r requirements.txt

# Usage

A call of

    python prepare-data.py <data.conf>

constructs a lexical grammar, supertags and splits the corpus as specified in ```<data.conf>```.
There are already some configuration files prepared for NeGra, Tiger and discontinuous Penn Treebank.
However, they require these corpora lying in ```./copora/```.
The script writes files to the folder specified in the ```filename``` field of configuration files.

    python training.py <data.conf> <model.conf>

trains a sequence tagger as specified in ```<model.conf>``` using the prepared data.
Existing configurations are ```bilstm-model.conf``` and ```bert-model.conf```.
During training, the script writes checkpoints to a newly created folder named after ```<data.conf>```, ```<model.conf>``` and the current date.
Training is monitored using tensorboard, also there are lists of losses and scores in ```loss.tsv```.
