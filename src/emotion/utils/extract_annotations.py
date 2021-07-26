import numpy as np
from ..config import Config


def extract_offsets(annotations: np.array):
    if len(annotations.shape) == 1:
        indices = []
        print(annotations)
        for idx, i in enumerate(annotations):
            if i != Config.BILSTM_CLASSES.get("O"):
                indices.append(i)
    else:
        raise f"The dimensions are not suitatble. Required 1 or two, but found {annotations.shape}"
    if len(indices) != 0:
        return np.min(indices), np.max(indices)
    else:
        return 0
