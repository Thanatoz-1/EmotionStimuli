__author__ = "Tushar Dhyani"

import os
import torch


class Config:
    """Configuration file for the project.
    In order to make this file dynamic, please change the inputs to this class from a file.
    """

    BERT_TYPE = "bert-base-uncased"
    BERT_CLASSIFICATION_PATH = "/media/thanoz/Extention/UniversityProjects/EmotionStimuli/data/emotion_classification"
    BERT_MAX_LEN = 65
    CLASSIFCATION_MAP = torch.load(
        os.path.join(BERT_CLASSIFICATION_PATH, "class_mapping.pt")
    )
