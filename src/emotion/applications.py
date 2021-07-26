__author__ = "Tushar Dhyani"

import token
from emotion import Classification
from emotion.baseline import model_experiencer, model_target, model_cue, model_cause
from emotion.baseline import get_embedding_model
from emotion.utils import bilstm_preprocessing, extract_offsets
import numpy as np

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
        self.cause = model_cause
        self.cue = model_cue
        self.exp = model_experiencer
        self.target = model_target
        self.embedding_model = get_embedding_model()

    def analyse(self, text):
        tokens = bilstm_preprocessing(text)
        embeddings = self.embedding_model(np.expand_dims(np.array(tokens), axis=0))

        cue = self.cue(embeddings)
        target = self.target(embeddings)
        cause = self.cause(embeddings)
        exp = self.exp(embeddings)

        print(cue.shape, cause.shape, exp.shape, target.shape)
        cause = extract_offsets(np.argmax(cause, axis=-1)[0])
        if cause == 0:
            cause = None
        else:
            tokens[cause[0] : cause[1]]

        cue = extract_offsets(np.argmax(cue, axis=-1)[0])
        if cue == 0:
            cue = 0
        else:
            tokens[cue[0] : cue[1]]
        target = extract_offsets(np.argmax(target, axis=-1)[0])
        if target == 0:
            target = 0
        else:
            tokens[target[0] : target[1]]

        exp = extract_offsets(np.argmax(exp, axis=-1)[0])
        if exp == 0:
            exp = 0
        else:
            tokens[exp[0] : exp[1]]

        output = {
            "text": text,
            "emotion": self.classification.predict_class(text),
            "roles": {
                "cause": cause,
                "cue": cue,
                "target": target,
                "experiencer": exp,
            },
        }

        return output
