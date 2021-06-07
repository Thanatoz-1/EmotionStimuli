__author__ = "Maximilian Wegge"
from ..utils import Dataset
import random


def sim_pred(dataset: Dataset) -> None:
    """Randomly generate IOB-tag sequences for testing purposes.
    These random predictions are written to the given object of class Dataset

    Args:
        dataset (Dataset): subset of the data
    """
    rand_seed = 10
    for inst in dataset.ReturnInst():
        for label in inst.ReturnLabels():
            tokens = inst.ReturnTokens()
            # tags are slightly weighted to mimic distribution in acutal data.
            tags = 60 * ["O"] + 15 * ["B"] + 25 * ["I"]
            random.seed(rand_seed)
            random.shuffle(tags)
            rand_seed += 1
            prediction = []
            for i in range(len(tokens)):
                prediction.append(tags[random.randint(0, len(tags) - 1)])
            inst.SetPred(label=label, annotation=prediction)
    return None
