__author__ = "Tushar Dhyani"

import os
import pickle


class Config:
    """Configuration file for the project.
    In order to make this file dynamic, please change the inputs to this class from a file.
    """

    BERT_TYPE = "bert-base-uncased"
    BERT_CLASSIFICATION_PATH = "/home/thanoz/emotion_weights/emotion_classification"
    BERT_MAX_LEN = 65
    with open(
        os.path.join(BERT_CLASSIFICATION_PATH, "class_mapping.pt"), "rb"
    ) as handle:
        CLASSIFCATION_MAP = pickle.load(handle)
    PATH_TO_WORDID = "/home/thanoz/emotion_weights/word2id.pt"
    with open(PATH_TO_WORDID, "rb") as handle:
        data = pickle.load(handle)
    WORD2ID = data["word2id"]
    ID2WORD = data["id2word"]
    BILSTM_MAXLEN = 100
    SEED = 2010
    EMBEDDING_WEIGHTS = "/home/thanoz/emotion_weights/embedding_backbone_weights.h5"
    CAUSE_MODEL_PATH = "/home/thanoz/emotion_weights/entire_dataset_cause_len65.h5"
    TARGET_MODEL_PATH = "/home/thanoz/emotion_weights/entire_dataset_target_len65.h5"
    CUE_MODEL_PATH = "/home/thanoz/emotion_weights/entire_dataset_cue_len65.h5"
    EXP_MODEL_PATH = "/home/thanoz/emotion_weights/entire_dataset_experiencer_len65.h5"
    BILSTM_CLASSES = {"O": 0, "B": 1, "I": 2, "PAD": 3}
    EPOCHS = 10
    MAX_VERBS = 10
