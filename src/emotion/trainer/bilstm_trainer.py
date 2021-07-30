__author__ = "Tushar Dhyani"

import json
from collections import Counter
import tensorflow as tf
from emotion.baseline.bilstm import get_model
from emotion.config import Config
from emotion.baseline import get_embedding_model
from emotion.evaluation import metric_for_bilstm
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

tf.random.set_seed(Config.SEED)


class Methods:
    """Methods class for training the models end to end. This iterates on single label of every corpus."""

    def __init__(self, path: str, dataset: list, label: str) -> None:
        """Methods for training the models.

        Args:
            path (str): Path to the datset json
            dataset (list): List of dataset names to be trained upon
            label (str): The label to be targetted for training. One of [Experiencer, Cause, Cue, Target]
        """
        # take config from the
        self.dataset = json.load(open(path, "r"))
        self.data = {}
        self.labels = {}
        self.embedding_model = get_embedding_model()
        # ids = []
        self.labels_dict = Config.BILSTM_CLASSES
        self.inverted_labels = list(self.labels_dict.keys())
        # datasets = ["reman", "eca", "gne", "emotion-stimulus", "electoral_tweets"]
        if type(dataset) == str:
            datasets = [dataset]
        elif type(dataset) == list:
            datasets = dataset

        self.get_label = label
        for datapoint in self.dataset:
            if datapoint.get("dataset") in datasets:
                if datapoint.get("annotations").get(self.get_label) is not None:
                    if len(datapoint.get("tokens")) <= Config.BILSTM_MAXLEN:
                        idx = datapoint.get("id")
                        self.data[idx] = datapoint.get("tokens")
                        # ids.append(datapoint.get("id"))
                        # labels.append([labels_dict.get(i) for i in datapoint.get("annotations").get(get_label)])
                        self.labels[idx] = [
                            self.labels_dict.get(i)
                            for i in datapoint.get("annotations").get(self.get_label)
                        ]
        print(f"Loaded {len(self.data)} datapoints and {len(self.labels)} labels")

    def split_data(self, data, splt: float = 0.8, random=False):
        """Split data into testing and training.
        Splits at <splt>%, but can be tuned using the argument.

        Args:
            data ([str]): data to perform split.
            splt (float, optional): % data split. Defaults to 0.8.
            random (bool, optional): sequential split or random split flag. Defaults to False.

        Returns:
            [type]: [description]
        """
        if random:
            splt = int(len(data) * splt)
            print(splt)
            return (np.array(data[:splt]), np.array(data[splt:]))
        else:
            trx, tx = train_test_split(data, train_size=splt, random_state=Config.SEED)
            return (trx, tx)

    def get_training_data(self):
        """Get the training data and preprocess it."""

        def prepo_string(
            text: list,
            word2id: dict = Config.WORD2ID,
            maxlen: int = Config.BILSTM_MAXLEN,
            unknown: str = "unk",
            padding: str = "pad",
        ):
            """Clean strings inputs"""
            tokens = [word2id.get(i.lower(), word2id.get(unknown)) for i in text]
            tokens = tokens + ([word2id.get(padding)] * (maxlen - len(tokens)))
            return tokens[:maxlen]

        def id2string(ids: list, ids2word: list = Config.ID2WORD, padding: str = "pad"):
            """Convert ids to strings"""
            res = [ids2word[i] for i in ids if ids2word != padding]
            return " ".join(res)

        # id2classes = sorted(list(set([j for i in labels for j in i])))
        id2classes = sorted(list(set([j for i in self.labels for j in self.labels[i]])))
        id2classes = id2classes + [3]

        # def train(self,X, Y, epochs):
        # train_x = np.asarray([prepo_string(i) for i in data])
        train_x, train_x_ids = [], []
        for idx in self.data:
            train_x.append(prepo_string(self.data[idx]))
            train_x_ids.append(idx)
        train_x = np.asarray(train_x)

        # train_y = [i[:max_len]+([3]*(max_len-len(i[:max_len]))) for i in labels]
        train_y, train_y_ids = [], []
        for idx in self.labels:
            train_y.append(
                self.labels[idx][: Config.BILSTM_MAXLEN]
                + (
                    [3]
                    * (
                        Config.BILSTM_MAXLEN
                        - len(self.labels[idx][: Config.BILSTM_MAXLEN])
                    )
                )
            )
            train_y_ids.append(idx)

        train_y = np.array(train_y)
        return train_x, train_y

    def trainer(self):
        """Train the model for given input and given target"""
        train_x, train_y = self.get_training_data()
        self.model = get_model()
        tr_x, t_x = self.split_data(train_x, random=False)
        tr_y, t_y = self.split_data(train_y, random=False)
        concat_tr_y = tf.keras.utils.to_categorical(tr_y)
        concat_t_y = tf.keras.utils.to_categorical(t_y)
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.02, reduction=tf.keras.losses.Reduction.SUM
            ),
            # metrics=["accuracy"],
        )
        train_emb = self.embedding_model.predict(tr_x)
        test_emb = self.embedding_model.predict(t_x)
        epochs = Config.EPOCHS
        best_f1 = 0
        best_prec = 0
        best_recall = 0
        for i in tqdm(range(epochs), total=epochs):
            hist = self.model.fit(
                train_emb,
                concat_tr_y,
                batch_size=32,
                epochs=1,
                verbose=1,
                validation_data=(test_emb, concat_t_y),
            )
            train_prec, train_rec, train_f1 = metric_for_bilstm(
                [tr_x, train_emb], tr_y, self.model
            )
            test_prec, test_rec, test_f1 = metric_for_bilstm(
                [t_x, test_emb], t_y, self.model
            )
            print(
                f"Training dataset results on epoch {i+1} are: {train_prec:0.3f} , {train_rec:0.3f}, {train_f1:0.3f}"
            )
            print(
                f"Testing dataset results on epoch {i+1} are: {test_prec:0.3f}, {test_rec:0.3f}, {test_f1:0.3f}"
            )
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_prec = test_prec
                best_recall = test_rec
                self.model.save_weights(f"bilstm_{self.get_label}.h5")

        print(f"The best validation score is : {best_f1}")
        return {"best_prec": best_prec, "best_recall": best_recall, "best_f1": best_f1}
