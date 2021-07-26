import numpy as np
from ..config import Config
import tensorflow as tf
from tensorflow import keras


def get_embedding_model(maxlen: int = Config.BILSTM_MAXLEN):
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=len(Config.WORD2ID),
        output_dim=100,
        weights=[np.zeros((len(Config.WORD2ID), 100))],
        input_length=maxlen,
        trainable=False,
        name="Embedding_layer",
    )

    inp_sent = tf.keras.Input((maxlen,), name="input_seq")
    emb = embedding_layer(inp_sent)
    model = tf.keras.models.Model(inp_sent, emb)
    model.load_weights(Config.EMBEDDING_WEIGHTS)
    return model
