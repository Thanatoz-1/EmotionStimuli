__author__ = "Tushar Dhyani, Maximilian Wegge"


def calc_jaccard_score(span_1: dict, span_2: dict) -> float:
    """Calculate the jaccard score of the provided spans of tags.
    Jaccard index = Intersection of argument list / Union of argument list

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    Args:
        span_1 (dict): first span of tags. Format:
        {IndexOfTokenInSentence:
                "tag",
        ...
        }
        span_2 (dict): second span of tags. Format:
        {IndexOfTokenInSentence:
                "tag",
        ...
        }
    Returns:
        float: jaccard score for the given spans.
    """
    s1 = set(span_1)
    s2 = set(span_2)
    j_union = s1.union(s2)
    j_intersection = 0
    for sent_index in j_union:
        if sent_index in span_1 and sent_index in span_2:
            if span_1[sent_index] == span_2[sent_index]:
                # both spans have the same tag at the same position.
                j_intersection += 1
            elif span_1[sent_index] in ["B", "I"] and span_2[sent_index] in ["B", "I"]:
                # both spans have either a 'B' or an 'I' tag are the same position.
                # Thus, both spans of 'BI' tags intersect at this position.
                j_intersection += 1
        else:
            pass
    js = float(j_intersection / len(j_union))
    return js


def calc_precision(tp: int, fp: int) -> float:
    """Calculate Precision.

    Args:
        tp (int): amount of TP.
        fp (int): amount of FP.

    Returns:
        float: precision for the given amounts of TP and FP.
    """

    if tp + fp != 0:
        precision = float(tp / (tp + fp))

    else:
        # prevent zero division error.
        precision = 0

    return precision


def calc_recall(tp: int, fn: int) -> float:
    """Calculate recall.

    Args:
        tp (int): amount of TP.
        fn (int): amount of FN.

    Returns:
        float: recall for the given amounts of TP and FN.
    """
    if tp + fn != 0:
        recall = float(tp / (tp + fn))
    else:
        # prevent zero division error.
        recall = 0

    return recall


def calc_fscore(prec: float, rec: float, beta: int = 0.5) -> float:
    """Calculate f-score.

    Args:
        prec (float): precision.
        rec (float): recall.
        beta (int): beta parameter.

    Returns:
        float: f-score for given precision and recall with given beta.
    """
    if ((beta * beta * prec) + rec) != 0:
        fscore = float(
            ((1 + (beta * beta)) * prec * rec) / ((beta * beta * prec) + rec)
        )
    else:
        # prevent zero division error.
        fscore = 0

    return fscore
