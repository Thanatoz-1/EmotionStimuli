class Eval:
    def __init__(self, y_true: list[str, int], y_pred: list[str, int]) -> None:
        """Evaluation class to calculate the scores of prediction of the entire batch.

        Args:
            y_true (list[str, int]): List of list containing the predictions in either string or int format
            y_pred (list[str, int]): List of list of predicted labels containing either string or int of the target labels.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def get_tp(self):
        pass


def jaccard_score(y_true: list[str, int], y_pred: list[str, int]):
    """Calculate the jaccard score of the provided sequence slice
    Jaccard index = $\frac{Intersection of argument list}{Union of argument list}$

    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    Args:
        y_true (list[str, int]): Sequence slice gold labels
        y_pred (list[str, int]): Sequence slice predicted labels
    """
    s1 = set(y_true)
    s2 = set(y_pred)
    ji = float(len(s1.intersection(s2)) / len(s1.union(s2)))
    return ji


def tp(y_true: list[int], y_pred: list[int]):
    """Python function to calculate values of True Positives

    Args:
        y_true (list[int]): List of int containing the gold labels
        y_pred (list[int]): List of int containing the predicted labels
    """
    tpv = 0
    for gld, prd in zip(y_true, y_pred):
        if gld == 1 and prd == 1:
            tpv += 1
    return tpv


def fp(y_true: list[int], y_pred: list[int]):
    """Python function for calculating False positives

    Args:
        y_true (list[int]): List of int containing the gold labels
        y_pred (list[int]): List of int containing predicted labels
    """
    fpv = 0
    for gld, prd in zip(y_true, y_pred):
        if gld == 0 and prd == 1:
            fpv += 1
    return fpv


def tn(y_true: list[int], y_pred: list[int]):
    """Python function for calculating True negatives

    Args:
        y_true (list[int]): List of int containing the gold labels
        y_pred (list[int]): List of int containing predicted labels
    """
    tnv = 0
    for gld, prd in zip(y_true, y_pred):
        if gld == 0 and prd == 0:
            tnv += 1
    return tnv


def fn(y_true: list[int], y_pred: list[int]):
    """Python function for calculating False negatives

    Args:
        y_true (list[int]): List of int containing the gold labels
        y_pred (list[int]): List of int containing predicted labels
    """
    fnv = 0
    for gld, prd in zip(y_true, y_pred):
        if gld == 1 and prd == 0:
            fnv += 1
    return fnv


def precision(y_true: list[int], y_pred: list[int]):
    """Python function for calculating precision

    Args:
        y_true (list[int]): List of int containing the gold labels
        y_pred (list[int]): List of int containing predicted labels
    """
    tpv = tp(y_true, y_pred)
    fpv = fp(y_true, y_pred)
    prec = tpv / (tpv + fpv)
    return prec


def recall(y_true: list[int], y_pred: list[int]):
    """Python function for calculating recall

    Args:
        y_true (list[int]): List of int containing the gold labels
        y_pred (list[int]): List of int containing predicted labels
    """
    tpv = tp(y_true, y_pred)
    fnv = fn(y_true, y_pred)
    recll = tpv / (tpv + fnv)
    return recll


def fscore(y_true: list[int], y_pred: list[int], beta: int = 1):
    """Python function for calculating recall
    Ref: https://en.wikipedia.org/wiki/F-score
    Args:
        y_true (list[int]): List of int containing the gold labels
        y_pred (list[int]): List of int containing predicted labels
        beta (int, optional): [description]. Defaults to 1.
    """
    pv = precision(y_true, y_pred)
    rv = recall(y_true, y_pred)
    f = (1 + (beta * beta)) * ((pv * rv) / ((beta * beta * pv) + recall))
    return f
