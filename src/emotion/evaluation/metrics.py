__author__ = "Maximilian Wegge"


def calc_jaccard_score(y_true: dict, y_pred: dict):
    """Calculate the jaccard score of the provided sequence slice
    Jaccard index = $\frac{Intersection of argument list}{Union of argument list}$

    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    Args:
        y_true (List[str, int]): Sequence slice gold labels
        y_pred (List[str, int]): Sequence slice predicted labels
    """
    s1 = set(y_true)
    s2 = set(y_pred)
    j_union = s1.union(s2)
    j_intersection = 0
    for i in j_union:
        if i in y_true and i in y_pred:
            if y_true[i] in ["B", "I"] and y_pred[i] in ["B", "I"]:
                # B and I count as intersection
                j_intersection += 1
            elif y_true[i] == y_pred[i]:
                # exact intersection
                j_intersection += 1
        else:
            pass
    js = float(j_intersection / len(j_union))
    return js


def calc_precision(tp, fp):
    """Python function for calculating precision

    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
    """
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    return precision


def calc_recall(tp, fn):
    """Python function for calculating recall

    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
    """
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    return recall


def calc_fscore(prec, rec, beta):
    """Python function for calculating recall
    Ref: https://en.wikipedia.org/wiki/F-score
    Args:
        y_true (List[int]): List of int containing the gold labels
        y_pred (List[int]): List of int containing predicted labels
        beta (int, optional): [description]. Defaults to 1.
    """
    if ((beta * beta * prec) + rec) != 0:
        fscore = ((1 + (beta * beta)) * prec * rec) / ((beta * beta * prec) + rec)
    else:
        fscore = 0
    return fscore
