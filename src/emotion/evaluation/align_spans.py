__author__ = "Maximilian Wegge"
import copy
from .metrics import calc_jaccard_score


def gen_poss_align(frm: list, to: list) -> dict:
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

    if len([span for span in gld_alignms if len(gld_alignms[span]) > 1]) == 0:
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
        gld_aligned = True

    else:

        gld_aligned = False
        gld_alignms, prd_alignms = perform_align_op(
            op=ops[0], from_alignms=gld_alignms, to_alignms=prd_alignms
        )

    if len([span for span in prd_alignms if len(prd_alignms[span]) > 1]) == 0:
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
            return align_spans(gld_alignms, prd_alignms, ops[1:])
        else:
            return align_spans(gld_alignms, prd_alignms, ops)


def perform_align_op(op: str, from_alignms: dict, to_alignms: dict):
    nxt_from_alignms = copy.deepcopy(from_alignms)
    nxt_to_alignms = copy.deepcopy(to_alignms)
    for (frm_idx, frm_tag) in from_alignms:
        alignm = from_alignms[(frm_idx, frm_tag)]

        if len(alignm) > 1:
            to_spans = [(idx, alignm[idx][0]) for idx in alignm]

            if op == "del_O":

                # delete alignments from gold-spans to prediction-spans of label 'O'
                # (if multiple possible gold- to prediction-alignments available)
                for to_span in alignm:
                    to_spn_tag = alignm[to_span][0]
                    if to_spn_tag == "O":

                        nxt_from_alignms[(frm_idx, frm_tag)].pop(to_span)
                        nxt_to_alignms[(to_span, to_spn_tag)].pop(frm_idx)

            elif op == "del_no_choice":
                # delete alignments if another gold-span has only one possible alignment to the same prediction-span
                # (only if this current span would have at least one alignment left after deletion)
                potential_del = set([])
                for other_span in from_alignms:
                    other_alignm = from_alignms[other_span]

                    to_span = [(idx, other_alignm[idx][0]) for idx in other_alignm]

                    if len(to_span) == 1:
                        if to_span[0] in to_spans:

                            potential_del.add(to_span[0])

                if potential_del and len(potential_del) != len(alignm):

                    for del_spn in potential_del:

                        nxt_from_alignms[(frm_idx, frm_tag)].pop(del_spn[0])
                        nxt_to_alignms[del_spn].pop(frm_idx)

                else:
                    pass

            elif frm_tag != "O":  # and op == "del_shortest_js"
                # only if the gold-span is BI: delete alignment with shorter intersection when mutliple available
                # or delete random alignment if intersection of both alignments is of same length

                js_scores = {}
                for to_span in alignm:
                    js = alignm[to_span][1]
                    if js in js_scores:
                        js_scores[js].append((to_span, alignm[to_span][0]))
                    else:
                        js_scores[js] = [(to_span, alignm[to_span][0])]

                min_js_spn = js_scores[min(js_scores)]
                if len(min_js_spn) == 1:
                    to_del = min_js_spn[0]
                    nxt_from_alignms[(frm_idx, frm_tag)].pop(to_del[0])
                    nxt_to_alignms[to_del].pop(frm_idx)
                else:
                    if op == "del_shortest_js":
                        pass
                    else:
                        to_del = min_js_spn[0]
                        nxt_from_alignms[(frm_idx, frm_tag)].pop(to_del[0])
                        nxt_to_alignms[to_del].pop(frm_idx)

            else:
                pass
        else:
            pass
    return nxt_from_alignms, nxt_to_alignms
