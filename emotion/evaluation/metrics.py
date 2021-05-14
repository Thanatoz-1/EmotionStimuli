def jaccard_score(y_true, y_pred):
    """Calculate the jaccard score of the provided sequence slice
    Jaccard index = $\frac{Intersection of argument list}{Union of argument list}$

    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    Args:
        y_true (List[str, int]): Sequence slice gold labels
        y_pred (List[str, int]): Sequence slice predicted labels
    """
    s1 = set(y_true)
    s2 = set(y_pred)
    ji = float(len(s1.intersection(s2)) / len(s1.union(s2)))
    return ji


def tp(y_true, y_pred):
    """Python function to calculate values of True Positives

    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing the predicted labels
    """
    tpv = 0
    for gld, prd in zip(y_true, y_pred):
        if gld == 1 and prd == 1:
            tpv += 1
    return tpv


def fp(y_true, y_pred):
    """Python function for calculating False positives

    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
    """
    fpv = 0
    for gld, prd in zip(y_true, y_pred):
        if gld == 0 and prd == 1:
            fpv += 1
    return fpv


def tn(y_true, y_pred):
    """Python function for calculating True negatives

    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
    """
    tnv = 0
    for gld, prd in zip(y_true, y_pred):
        if gld == 0 and prd == 0:
            tnv += 1
    return tnv


def fn(y_true, y_pred):
    """Python function for calculating False negatives

    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
    """
    fnv = 0
    for gld, prd in zip(y_true, y_pred):
        if gld == 1 and prd == 0:
            fnv += 1
    return fnv


def precision(y_true, y_pred):
    """Python function for calculating precision

    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
    """
    tpv = tp(y_true, y_pred)
    fpv = fp(y_true, y_pred)
    prec = tpv / (tpv + fpv)
    return prec


def recall(y_true, y_pred):
    """Python function for calculating recall

    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
    """
    tpv = tp(y_true, y_pred)
    fnv = fn(y_true, y_pred)
    recll = tpv / (tpv + fnv)
    return recll


def fscore(y_true, y_pred, beta: int = 1):
    """Python function for calculating recall
    Ref: https://en.wikipedia.org/wiki/F-score
    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
        beta (int, optional): [description]. Defaults to 1.
    """
    pv = precision(y_true, y_pred)
    rv = recall(y_true, y_pred)
    f = (1 + (beta * beta)) * ((pv * rv) / ((beta * beta * pv) + recall))
    return f
