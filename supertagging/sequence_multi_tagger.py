from flair.models.sequence_tagger_model import *

class SequenceMultiTagger(SequenceTagger):
    def __init__(self, hidden_size, embeddings, tag_dictionaries, tag_types, **kwargs):
        # exclude args that don't make sense anymore
        # TODO: the first two are just due to the way we instantiate this class
        assert "hidden_size" not in kwargs
        assert "embeddings" not in kwargs
        assert "tag_dictionary" not in kwargs
        assert "tag_types" not in kwargs
        assert "loss_weights" not in kwargs

        super(SequenceMultiTagger, self).__init__(hidden_size, embeddings, tag_dictionaries[0], tag_types[0], **kwargs)
        
        self.tag_dictionaries = tag_dictionaries
        if self.use_crf:
            for tag_dictionary in self.tag_dictionaries[1:]:
                tag_dictionary.add_item(START_TAG)
                tag_dictionary.add_item(STOP_TAG)

        self.tag_types = tag_types

        self.tagset_sizes = [len(tag_dictionary) for tag_dictionary in self.tag_dictionaries]

        self.weight_dict = self.loss_weights = None

        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1
            self.linear_layers = [
                torch.nn.Linear(hidden_size * num_directions, len(tag_dictionary))
                for tag_dictionary in self.tag_dictionaries
            ]
        else:
            self.linear_layers = [
                torch.nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))
                for tag_dictionary in self.tag_dictionaries
            ]
        for linear in self.linear_layers:
            linear.to(flair.device)

        # TODO crf

        self.to(flair.device)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "tag_dictionaries": self.tag_dictionaries,
            "tag_types": self.tag_types,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "reproject_embeddings": self.reproject_embeddings,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if "use_locked_dropout" not in state.keys()
            else state["use_locked_dropout"]
        )
        train_initial_hidden_state = (
            False
            if "train_initial_hidden_state" not in state.keys()
            else state["train_initial_hidden_state"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        # TODO!
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        reproject_embeddings = True if "reproject_embeddings" not in state.keys() else state["reproject_embeddings"]
        if "reproject_to" in state.keys():
            reproject_embeddings = state["reproject_to"]

        model = SequenceMultiTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionaries=state["tag_dictionaries"],
            tag_types=state["tag_types"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            train_initial_hidden_state=train_initial_hidden_state,
            rnn_type=rnn_type,
            beta=beta,
            loss_weights=weights,
            reproject_embeddings=reproject_embeddings,
        )
        model.load_state_dict(state["state_dict"])
        return model

    def predict(*args, **kwargs):
        raise NotImplemented()

    def evaluate(*args, **kwargs):
        raise NotImplemented()

    def forward_loss(
        self,
        data_points: Union[List[Sentence], Sentence],
        sort=True
    ) -> torch.tensor:
        loss = 0
        features = tuple(self.forward(data_points))
        for i, (tag_dictionary, tag_type, tagset_size, linear) in enumerate(
            zip(
                self.tag_dictionaries,
                self.tag_types,
                self.tagset_sizes,
                self.linear_layers
            )
        ):
            self.tag_dictionary = tag_dictionary
            self.tag_type = tag_type
            self.tagset_size = tagset_size
            self.linear = linear
            loss += self._calculate_loss(features[i], data_points)
        return loss

    def forward(self, sentences: List[Sentence]):
        for i, (tag_dictionary, tag_type, tagset_size, linear) in enumerate(
            zip(
                self.tag_dictionaries,
                self.tag_types,
                self.tagset_sizes,
                self.linear_layers
            )
        ):
            self.tag_dictionary = tag_dictionary
            self.tag_type = tag_type
            self.tagset_size = tagset_size
            self.linear = linear
            yield super(SequenceMultiTagger, self).forward(sentences)
