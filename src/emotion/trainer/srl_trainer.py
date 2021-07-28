__author__ = "Tushar Dhyani"

import json
from collections import Counter
from scipy.sparse.construct import rand
import tensorflow as tf
from emotion.baseline.srl import get_srl_model
from emotion.config import Config
from emotion.baseline import get_embedding_model
from sklearn.model_selection import train_test_split
from emotion.evaluation import metric_for_bilstm
import numpy as np
from tqdm import tqdm

tf.random.set_seed(Config.SEED)


def pad_srl(feature, maxverbs: int = 10, maxlen: int = Config.BILSTM_MAXLEN):
    padded_feature = np.ones((10, maxlen)) * 27
    for idx, i in enumerate(feature[:maxverbs]):
        for jdx, j in enumerate(i[:maxlen]):
            padded_feature[idx][jdx] = j
    return padded_feature


class Methods:
    def __init__(
        self, dataset_path: str, srl_path: str, dataset_name: list, label_name: str
    ) -> None:
        # take config from the
        self.dataset = json.load(open(dataset_path, "r"))
        self.srl_features_all = json.load(open(srl_path, "r"))
        self.data = {}
        self.labels = {}
        self.embedding_model = get_embedding_model()
        # ids = []
        self.labels_dict = Config.BILSTM_CLASSES
        self.inverted_labels = list(self.labels_dict.keys())
        # datasets = ["reman", "eca", "gne", "emotion-stimulus", "electoral_tweets"]
        if type(dataset_name) == str:
            datasets = [dataset_name]
        elif type(dataset_name) == list:
            datasets = dataset_name

        self.get_label = label_name
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
        self.srl = []
        for idx in self.data:
            features = self.srl_features_all[idx]
            self.srl.append(pad_srl(features))
        self.srl = np.array(self.srl)
        print(f"Loaded {len(self.data)} datapoints and {len(self.labels)} labels")

    def split_data(self, data, splt: float = 0.8, random=False):
        if random:
            splt = int(len(data) * splt)
            print(splt)
            return (np.array(data[:splt]), np.array(data[splt:]))
        else:
            trx, tx = train_test_split(data, train_size=splt, random_state=Config.SEED)
            return (trx, tx)

    def get_training_data(self):
        def prepo_string(
            text: list,
            word2id: dict = Config.WORD2ID,
            maxlen: int = Config.BILSTM_MAXLEN,
            unknown: str = "unk",
            padding: str = "pad",
        ):
            tokens = [word2id.get(i.lower(), word2id.get(unknown)) for i in text]
            tokens = tokens + ([word2id.get(padding)] * (maxlen - len(tokens)))
            return tokens[:maxlen]

        def id2string(ids: list, ids2word: list = Config.ID2WORD, padding: str = "pad"):
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
        concat_tr_y = np.concatenate(
            (train_y, np.ones((train_y.shape[0], 10)) * 3), axis=1
        )
        return train_x, concat_tr_y

    def trainer(self):
        train_x, train_y = self.get_training_data()
        self.model = get_srl_model()
        tr_x, t_x = self.split_data(train_x, random=True)
        tr_srl, t_srl = self.split_data(self.srl, random=True)
        tr_y, t_y = self.split_data(train_y, random=True)
        concat_tr_y = tf.keras.utils.to_categorical(tr_y)
        concat_t_y = tf.keras.utils.to_categorical(t_y)
        print("Shape of x: ", tr_x.shape, t_x.shape)
        print("Shape of srl: ", tr_srl.shape, t_srl.shape)
        print("Shape of y: ", tr_y.shape, t_y.shape)
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
        best_score = 0
        for i in tqdm(range(epochs), total=epochs):
            hist = self.model.fit(
                [train_emb, tr_srl],
                concat_tr_y,
                batch_size=32,
                epochs=1,
                verbose=1,
                validation_data=([test_emb, t_srl], concat_t_y),
            )
            train_prec, train_rec, train_f1 = metric_for_bilstm(
                [tr_x, train_emb, tr_srl], tr_y, self.model, contains_srl=True
            )
            test_prec, test_rec, test_f1 = metric_for_bilstm(
                [t_x, test_emb, t_srl], t_y, self.model, contains_srl=True
            )
            print(
                f"Training dataset results on epoch {i+1} are: {train_prec:0.3f} , {train_rec:0.3f}, {train_f1:0.3f}"
            )
            print(
                f"Testing dataset results on epoch {i+1} are: {test_prec:0.3f}, {test_rec:0.3f}, {test_f1:0.3f}"
            )
            if test_f1 > best_score:
                best_score = test_f1
                self.model.save_weights("bilstm_role.h5")

        print(f"The best validation score is : {best_score}")


if __name__ == "__main__":
    tst = Methods(
        "/media/thanoz/Extention/UniversityProjects/EmotionStimuli/data/rectified-unified-with-offsets.json",
        "reman",
        "cause",
    )
    x, y = tst.get_training_data()
    print(x.shape, y.shape)
