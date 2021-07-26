import tensorflow as tf
from tensorflow import keras
from ..config import Config


def get_model(entity, maxlen: int = Config.BILSTM_MAXLEN, feat_len: int = 100):
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
    model.load_weights(entity)
    # print(model.summary())
    return model


model_cause = get_model(entity=Config.CAUSE_MODEL_PATH)
model_target = get_model(entity=Config.TARGET_MODEL_PATH)
model_cue = get_model(entity=Config.CUE_MODEL_PATH)
model_experiencer = get_model(entity=Config.EXP_MODEL_PATH)
