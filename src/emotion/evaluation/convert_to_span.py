__author__ = "Maximilian Wegge"

from typing import List
from .align_spans import gen_poss_align, align_spans


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


def conv2brown(tokens, labels):
    """converts list of tokens and list of corresponding
    labels into Brown Format"""
    brown = []
    for tok, lab in zip(tokens, labels):
        brown.append((tok.lower(), lab))
    return brown


def get_counts(gold, pred, threshold: int = 0.8, return_indices=False):

    tp = 0
    fp = 0
    fn = 0
    indices = []
    # get the spans of tags in the gold and the predicted annotations.
    gld_spn = conv2span(gold)
    prd_spn = conv2span(pred)

    # generate all possible alignment__g2ps between the spans.
    poss_g2p = gen_poss_align(frm=gld_spn, to=prd_spn)
    poss_p2g = gen_poss_align(frm=prd_spn, to=gld_spn)

    # delete ambiguous alignment__g2ps to obtain final alignment__g2p
    final_g2p = align_spans(
        poss_g2p,
        poss_p2g,
        ops=["del_O", "del_no_choice", "del_shortest_jaccard", "del_rand"],
    )

    for idx, gold_alignm in enumerate(final_g2p):

        for pred_span in final_g2p[gold_alignm]:

            pred_alignm = final_g2p[gold_alignm][pred_span]

            jaccard = pred_alignm[1]
            pred_tag = pred_alignm[0]

            if jaccard == 0:
                if pred_tag == "O":
                    # evaluates to FN.
                    fn += 1

                else:
                    # evaluates to FP.
                    fp += 1

            elif jaccard > 0 and jaccard < threshold:
                if pred_tag == "O":
                    # would evaluate to TN if jaccard-score would have
                    # been above threshold.
                    pass
                else:
                    # would evalaute to TP if jaccard-score would have
                    # been above threshold. Thus, evaluates to FN.
                    # OR might be omitted.
                    fn += 1
                    pass
            else:
                if pred_tag == ".":
                    # full stops are not evaluated.
                    pass
                elif pred_tag == "O":
                    # evaluates to TN but is omitted here as TN are not
                    # relevant for calculating precision, recall and f-score.
                    pass
                else:
                    # evaluates to TP.
                    tp += 1
                    indices.append(idx)

    if return_indices:
        return (tp, fp, fn), indices
    return (tp, fp, fn)
