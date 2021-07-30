__author__ = "Tushar Dhyani"

import tensorflow as tf
from tensorflow import keras
from ..config import Config

tf.random.set_seed(Config.SEED)


def get_model(entity=None, maxlen: int = Config.BILSTM_MAXLEN, feat_len: int = 100):
    """BiLSTM model with glove embedding as feature vector input.
    This model is used for training and inferences in the framework.

    Args:
        entity (str, optional): The path of the saved models that you want to load.
                                Defaults to None.
        maxlen (int, optional): Maximum length of the sequence. This has can be changed from Config file.
                                Defaults to Config.BILSTM_MAXLEN.
        feat_len (int, optional): Dimension of glove feature vector. Defaults to 100.

    Returns:
        tensorflow.Model: Returns the trainable model which could be trained using Keras as well as Tensorflow.
    """
    inp_sent = tf.keras.Input((maxlen, feat_len), name="input_embedding")
    x = tf.keras.layers.Conv1D(300, 3, activation="relu", name="CNN1", padding="same")(
        inp_sent
    )
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=100, return_sequences=True, recurrent_dropout=0.5),
        name="Bidir1",
    )(x)
    intermediate = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=100, return_sequences=True, recurrent_dropout=0.5),
        name="Bidir2",
    )(x)
    x = tf.keras.layers.Dense(len(Config.BILSTM_CLASSES), activation="softmax")(
        intermediate
    )

    model = tf.keras.models.Model(inp_sent, x)
    if entity != None:
        print(f"Model weights loaded from {entity}")
        model.load_weights(entity)
    # print(model.summary())
    return model


model_cause = get_model(entity=Config.CAUSE_MODEL_PATH)
model_target = get_model(entity=Config.TARGET_MODEL_PATH)
model_cue = get_model(entity=Config.CUE_MODEL_PATH)
model_experiencer = get_model(entity=Config.EXP_MODEL_PATH)
