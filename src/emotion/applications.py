__author__ = "Tushar Dhyani"

from emotion import Classification

# Create 3 important usecases for the repo.
# 1. For Emotion role labelling
# 2. For emotion classification
# 3. For emotion analysis


class EmotionRoleLabeller:
    def __init__(self) -> None:
        # init role classifier
        # init role labeller
        # Create predictor
        self.classification = Classification()
        pass

    def analyse(self, text):
        output = {
            "text": text,
            "emotion": self.classification.predict_class(text),
            "roles": None,
        }

        return output
