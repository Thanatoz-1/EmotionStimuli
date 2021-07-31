__author__ = "Tushar Dhyani"

from transformers import BertForSequenceClassification
from ..config import Config
from ..utils import bert_preprocessing
import torch


class Classification:
    """Emotion classification class for predicting emotions based on the dataset.
    This class is primarily reponsible for predicting emotions.
    """

    def __init__(self) -> None:
        """The init of this file is controlled from Config class in the config package.
        Please change configurations in the Config class to make changes to this class.
        """
        print(f"Loading {Config.BERT_CLASSIFICATION_PATH}")
        self.model = BertForSequenceClassification.from_pretrained(
            Config.BERT_CLASSIFICATION_PATH
        )
        self.class2id = Config.CLASSIFCATION_MAP["classids"]
        self.id2class = Config.CLASSIFCATION_MAP["ids2class"]
        self.model.eval()

    def predict_class(self, text: str) -> str:
        """Major function for predicting the classification on a
        given piece of text.

        Args:
            text ([str]): Text for classifying the emotion.

        Returns:
            [str]: The respecitve emotion from the class2ids for that particular string.
        """
        tt = bert_preprocessing(text)
        ttxt = {k: torch.tensor(v).unsqueeze(0) for k, v in tt.items()}
        with torch.no_grad():
            output = self.model(**ttxt)
        predictions = output.logits.argmax(-1)
        return self.id2class[predictions.item()]
