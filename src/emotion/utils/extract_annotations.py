__author__ = "Tushar Dhyani"

import numpy as np
from ..config import Config


def extract_offsets(annotations: np.array):
    """Extract the offsets from a prediction

    Args:
        annotations (np.array): The array of predictions from a model after argmaxed

    Raises:
        f: Dimensions unsuitable if argmax is not taken

    Returns:
        [minimum, maximum]
    """
    if len(annotations.shape) == 1:
        indices = []
        for idx, i in enumerate(annotations):
            if i != Config.BILSTM_CLASSES.get("O"):
                indices.append(i)
    else:
        raise f"The dimensions are not suitable. Required 1 or two, but found {annotations.shape}"
    if len(indices) != 0:
        return np.min(indices), np.max(indices)
    else:
        return 0
