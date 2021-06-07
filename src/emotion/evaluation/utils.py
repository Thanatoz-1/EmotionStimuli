import copy


def gen_poss_align(frm: list, to: list) -> dict:
    from_sets = [set(span) for span in frm]
    to_sets = [set(span) for span in to]
    poss_align = {}
    for i in range(len(from_sets)):
        poss_align[i] = {}
        for j in range(len(to_sets)):
            intrsct = from_sets[i].intersection(to_sets[j])
            if len(intrsct) > 0:
                poss_align[i][j] = (list(to[j].values())[0], len(intrsct))
    return poss_align


def align_spans(
    gld_alignms: dict, prd_alignms: dict, ops=["delO", "no-choice", "intrsct"]
) -> dict:
    # nxt_gld = copy.deepcopy(gld_alignms)
    # nxt_prd = copy.deepcopy(prd_alignms)

    if len([span for span in gld_alignms if len(gld_alignms[span]) > 1]) == 0:
        gld_aligned = True
    else:
        gld_aligned = False
        gld_alignms, prd_alignms = perform_align_op(
            op=ops[0], from_alignms=gld_alignms, to_alignms=prd_alignms
        )

    if len([span for span in prd_alignms if len(prd_alignms[span]) > 1]) == 0:
        prd_aligned = True
    else:
        prd_aligned = False
        prd_alignms, gld_alignms = perform_align_op(
            op=ops[0], from_alignms=prd_alignms, to_alignms=gld_alignms
        )

    if (gld_aligned and prd_aligned) or len(ops) == 1:
        return gld_alignms
    else:
        ops.pop(0)
        return align_spans(gld_alignms, prd_alignms, ops)


def perform_align_op(op: str, from_alignms: dict, to_alignms: dict):
    nxt_from_alignms = copy.deepcopy(from_alignms)
    nxt_to_alignms = copy.deepcopy(to_alignms)

    for from_span in from_alignms:
        alignm = from_alignms[from_span]

        if len(alignm) > 1:
            to_spans = list(alignm.keys())
            from_spn_tag = to_alignms[to_spans[0]][from_span][0]

            if op == "delO":
                # delete alignments from gold-spans to prediction-spans of label 'O'
                # (if multiple possible gold- to prediction-alignments available)
                for to_span in alignm:
                    to_spn_tag = alignm[to_span][0]
                    if to_spn_tag == "O":
                        nxt_from_alignms[from_span].pop(to_span)
                        nxt_to_alignms[to_span].pop(from_span)

            elif op == "no-choice":
                # delete alignments if another gold-span has only one possible alignment to the same prediction-span
                for other_span in from_alignms:
                    other_alignm = from_alignms[other_span]
                    if len(other_alignm) == 1:
                        to_span = list(other_alignm.keys())[0]
                        if to_span in to_spans:
                            nxt_from_alignms[from_span].pop(to_span)
                            nxt_to_alignms[to_span].pop(from_span)

            elif op[0] == "intrsct" and from_spn_tag != "O":
                # only if the gold-span is BI: delete alignment with shorter intersection when mutliple available
                # or delete random alignment if intersection of both alignments is of same length
                # intrscts = [alignm[span][1] for span in alignm]
                # intrscts = dict(sorted(alignm, key = lambda item: alignm[item][1]))
                # min_intrsct_spn = sorted(alignm, key = lambda item: alignm[item][1])[0]

                intrscts = {}
                for to_span in alignm:
                    intrsct = alignm[to_span][1]
                    if intrsct in intrscts:
                        intrscts[intrsct].append(to_span)
                    else:
                        intrscts[intrsct] = [to_span]

                min_intrsct_spn = intrscts(min(intrscts))

                if len(min_intrsct_spn) < 1:
                    # delete random
                    random.shuffle(min_intrsct_spn)

                else:
                    # delete shortest
                    pass

                nxt_from_alignms[from_span].pop(min_intrsct_spn[0])
                nxt_to_alignms[min_intrsct_spn[0]].pop(from_span)

            else:
                pass
        else:
            pass

    return nxt_from_alignms, nxt_to_alignms
