__author__ = "Maximilian Wegge"
import copy, random
from typing import Tuple
from .metrics import calc_jaccard_score


def gen_poss_align(frm: list, to: list) -> dict:
    """Generate all possible alignments between the gold IOB-tag-spans
    and the predicted IOB-tag-spans in a sentence. An alignment is possible
    if the intersection of two spans contains at least one token. The alignment
    is directed, from one list of spans to another.

    Args:
        frm (list): specifies the direction for the generation of
        possible alignments (from this list).
        to (list): specifies the direction for the generation of
        possible alignments (to this list).

    Returns:
        dict: all possible alignments in the specified direction.
        {
            # tuple(index of span in from-list, first tag of span in from-list)
            (0, 'B'):
                {
                    # index of aligned span in to-list
                    0:
                        # tuple(first tag of aligned span in to-list,
                        # jaccard-score for the two spans)
                        ('B', 0.3333333333333333)
                },
            (2, 'B'): {
                1: ('B', 1.0)
                }
        }
    """

    poss_align = {}
    for i in range(len(frm)):
        idx_and_tag = (i, list(frm[i].values())[0])
        poss_align[idx_and_tag] = {}
        for j in range(len(to)):
            intrsct = set(frm[i]).intersection(set(to[j]))
            if len(intrsct) > 0:
                poss_align[idx_and_tag][j] = (
                    list(to[j].values())[0],
                    calc_jaccard_score(frm[i], to[j]),
                )
    return poss_align


def align_spans(
    gld_alignms: dict,
    prd_alignms: dict,
    ops=["del_O", "del_no_choice", "del_shortest_js", "del_rand"],
) -> dict:
    """Follow fixed heuristic to eliminate ambiguous alignments (more than
    one possible alignment from one span to another). Exception: One span
    of 'O'-tags can be aligned to more than one span of 'BI'-tags, to catch
    every FP/FN. Recursive.

    Args:
        gld_alignms (dict): (all possible) alignments, directed from gold to
        predicted spans.
        prd_alignms (dict): (all possible) alignments, directed from predicted
        to gold spans.
        ops (list, optional): Heuristic steps that need to be performed to
        eliminate ambiguous alignments. Defaults to
        ["del_O", "del_no_choice", "del_shortest_js", "del_rand"].

    Returns:
        dict: unambiguous alignment of gold spans and predicted spans.
    """

    if len([span for span in gld_alignms if len(gld_alignms[span]) > 1]) == 0:
        # each span in the gold annotation is aligned to at most one span
        # in the predicted annotation.
        gld_aligned = True

    elif (
        ops[0] != "del_O"
        and len(
            [
                span
                for span in gld_alignms
                if span[1] != "O" and len(gld_alignms[span]) > 1
            ]
        )
        == 0
    ):
        # each span of tags 'BI' in the gold annotation is aligned to at
        # most one span in the predicted annotation, but not all spans of
        # tags 'O' are. However, this alignment is considered aligned
        # (if the first operation of the heuristic has been performed).
        gld_aligned = True

    else:

        # if not all spans of tags 'BI' are aligned, the next step in the heuristic
        # is performed.
        gld_aligned = False
        gld_alignms, prd_alignms = perform_align_op(
            op=ops[0], from_alignms=gld_alignms, to_alignms=prd_alignms
        )

    if len([span for span in prd_alignms if len(prd_alignms[span]) > 1]) == 0:
        # each span in the predicted annotation is aligned to at most one span
        # in the gold annotation.
        prd_aligned = True
    elif (
        ops[0] != "del_O"
        and len(
            [
                span
                for span in prd_alignms
                if span[1] != "O" and len(prd_alignms[span]) > 1
            ]
        )
        == 0
    ):
        # each span of tags 'BI' in the predicted annotation is aligned to
        # at most one span in the gold annotation, but not all spans of
        # tags 'O' are. However, this alignment is considered fully aligned
        # (if the first operation of the heuristic has been performed).
        prd_aligned = True
    else:
        prd_aligned = False
        prd_alignms, gld_alignms = perform_align_op(
            op=ops[0], from_alignms=prd_alignms, to_alignms=gld_alignms
        )

    if gld_aligned and prd_aligned:
        return gld_alignms
    else:
        if len(ops) != 1:
            # call align_spans with next step in heuristic
            return align_spans(gld_alignms, prd_alignms, ops[1:])
        else:
            # the last step of the heuristic is performed until
            # all spans are aligned.
            return align_spans(gld_alignms, prd_alignms, ops)


def perform_align_op(
    op: str, from_alignms: dict, to_alignms: dict
) -> Tuple[dict, dict]:
    """Perform step in heuristic to eliminate ambiguous alignments
    between spans.

    Args:
        op (str): current step in heuristic that is performed.
        from_alignms (dict): alignments between spans of annotated
        IOB-tags, directed from annotation a to annotation b.
        to_alignms (dict): alignments between spans of annotated
        IOB-tags, directed from annotation b to annotation a.

    Returns:
        tuple(dict): alignments between spans of annotated IOB-tags
        after the heuristic step has been performed (for both directions).
    """

    nxt_from_alignms = copy.deepcopy(from_alignms)
    nxt_to_alignms = copy.deepcopy(to_alignms)

    for (frm_i, frm_tag) in from_alignms:
        alignm = from_alignms[(frm_i, frm_tag)]

        if len(alignm) > 1:
            to_spans = [(i, alignm[i][0]) for i in alignm]

            if op == "del_O":
                # delete alignment from span of any tag to
                # spans of tag 'O' (only if current alignment
                # is ambiguous)

                for to_span in alignm:
                    to_spn_tag = alignm[to_span][0]
                    if to_spn_tag == "O":

                        nxt_from_alignms[(frm_i, frm_tag)].pop(to_span)
                        nxt_to_alignms[(to_span, to_spn_tag)].pop(frm_i)

            elif op == "del_no_choice":
                # delete alignments to to-span if another from-span has
                # an unambiguous alignment to the same to-span.
                # Only perform this step if this current span would have
                # at least one alignment left after deleting the alignments.
                # If this would not be the case, do not delete any alignments
                # to prevent randomly aligned spans in this step.
                potential_del = set([])
                for other_span in from_alignms:
                    other_alignm = from_alignms[other_span]

                    to_span = [(i, other_alignm[i][0]) for i in other_alignm]

                    if len(to_span) == 1:
                        if to_span[0] in to_spans:

                            potential_del.add(to_span[0])

                if potential_del and len(potential_del) != len(alignm):

                    for del_spn in potential_del:

                        nxt_from_alignms[(frm_i, frm_tag)].pop(del_spn[0])
                        nxt_to_alignms[del_spn].pop(frm_i)

                else:
                    pass

            elif frm_tag != "O":
                # The last two steps in the heuristic do only apply
                # to from-spans of tags 'BI'.

                js_scores = {}
                for to_span in alignm:
                    js = alignm[to_span][1]
                    if js in js_scores:
                        js_scores[js].append((to_span, alignm[to_span][0]))
                    else:
                        js_scores[js] = [(to_span, alignm[to_span][0])]

                min_js_spn = js_scores[min(js_scores)]
                if len(min_js_spn) == 1:
                    # delete the alignment with the lowest jaccard score,
                    # if there is only one.
                    to_del = min_js_spn[0]
                    nxt_from_alignms[(frm_i, frm_tag)].pop(to_del[0])
                    nxt_to_alignms[to_del].pop(frm_i)
                else:
                    if op == "del_shortest_js":
                        # if multiple alignments share the same (lowest)
                        # jaccard score, do not delete any alignments to
                        # prevent randomly aligned spans in this step.
                        pass

                    elif op == "del_rand":
                        # if there are ambiguous alignments that could
                        # not be eliminated in the previous steps of
                        # the heuristic, an alignment is selected
                        # randomly for deletion.
                        random.seed(11)
                        random.shuffle(min_js_spn)
                        to_del = min_js_spn[0]
                        nxt_from_alignms[(frm_i, frm_tag)].pop(to_del[0])
                        nxt_to_alignms[to_del].pop(frm_i)

            else:
                pass
        else:
            pass
    return nxt_from_alignms, nxt_to_alignms
