__author__ = "Tushar Dhyani, Maximilian Wegge"

import tensorflow as tf
from tensorflow import keras
from ..config import Config

tf.random.set_seed(Config.SEED)


def get_srl_model(
    entity=None,
    maxlen_sent: int = Config.BILSTM_MAXLEN,
    maxlen_srl: int = Config.MAX_VERBS,
    featrs: int = 100,
):
    """Generates the SRL model for training and inferencing.

    Args:
        entity (str, optional): Preloaded model path for a particular entity. Defaults to None.
        maxlen_sent (int, optional): Please tweak this from Config. Defaults to Config.BILSTM_MAXLEN.
        maxlen_srl (int, optional): Please tweak this from Config. Defaults to Config.MAX_VERBS.
        featrs (int, optional): length of features. Defaults to 100.

    Returns:
        tensorflow.Model: Returns the tensorflow model compatible with Keras and Tensorflow.
    """

    inp_sent = tf.keras.Input((maxlen_sent, featrs), name="input_seq")
    inp_srl = tf.keras.Input((maxlen_srl, featrs), name="input_srl")

    concat = tf.keras.layers.Concatenate(axis=1)([inp_sent, inp_srl])

    x = tf.keras.layers.Conv1D(
        300, 5, activation="relu", name="CNN1", strides=1, padding="same"
    )(concat)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=100, return_sequences=True, recurrent_dropout=0.5, name="LSTM1"
        ),
        name="Bidir1",
    )(x)
    intermediate = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=100, return_sequences=True, recurrent_dropout=0.5, name="LSTM2"
        ),
        name="Bidir2",
    )(x)
    x = tf.keras.layers.Dense(len(Config.BILSTM_CLASSES), activation="softmax")(
        intermediate
    )

    model = tf.keras.models.Model([inp_sent, inp_srl], x)
    return model
