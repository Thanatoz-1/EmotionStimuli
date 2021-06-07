__author__ = "Maximilian Wegge"
from typing import List


def conv2span(annot_brown: list) -> List[dict]:
    """Extract the spans (i.e. consecutive sequences of either
    'O'- or 'B'/'I'-tags) from a sentences's annotation.

    Args:
        annot_brown (list): list of annotations in brown format:
        [
            ("token", "tag"),
            ...
        ]

    Returns:
        [type]: list of spans (each span is a dictionary):
        [
            {IndexOfTokenInSentence:
                "tag"
            },
            ...
        ]
    """
    all_spans = []
    current_span = {}
    iob_tags = [iob_tag for (token, iob_tag) in annot_brown]

    for i in range(len(iob_tags)):
        iob_tag = iob_tags[i]
        if i == 0:
            current_span[i] = iob_tag
        elif iob_tag == "I" and current_span[i - 1] in ["B", "I"]:
            current_span[i] = iob_tag
        elif iob_tag == "O" and current_span[i - 1] == "O":
            current_span[i] = iob_tag
        else:
            all_spans.append(current_span)
            current_span = {}
            current_span[i] = iob_tag

        if i == len(iob_tags) - 1:
            all_spans.append(current_span)
        else:
            pass

    return all_spans
