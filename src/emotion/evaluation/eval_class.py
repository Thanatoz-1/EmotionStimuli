from .metrics import calc_precision, calc_recall, calc_fscore, calc_jaccard_score
from .utils import gen_poss_align, align_spans
from ..utils.utils import conv2span


class Evaluation:
    def __init__(self, dataset, threshold: int = 0.8, beta=1, ib=True) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.documentation = {}
        self.evaluate(dataset=dataset, threshold=threshold, ib=ib)
        self.precision = calc_precision(tp=self.tp, fp=self.fp)
        self.recall = calc_recall(tp=self.tp, fn=self.fn)
        self.fscore = calc_fscore(prec=self.precision, rec=self.recall, beta=beta)

    def evaluate(self, dataset, threshold, ib):
        for id in dataset.instances:

            self.documentation[id] = {}
            self.documentation[id]["tokens"] = dataset.instances[id].tokens
            self.documentation[id]["annotations"] = {}
            self.documentation[id]["spans"] = {}

            for label in dataset.instances[id].labels:

                self.documentation[id]["annotations"][label] = (
                    dataset.instances[id].gold[label],
                    dataset.instances[id].pred[label],
                )

                gld_spn = conv2span(dataset.instances[id].gold[label])
                prd_spn = conv2span(dataset.instances[id].pred[label])

                self.documentation[id]["spans"][label] = (
                    gld_spn,
                    prd_spn,
                )

                poss_g2p = gen_poss_align(frm=gld_spn, to=prd_spn)
                poss_p2g = gen_poss_align(frm=prd_spn, to=gld_spn)
                # print()
                # print("#####Funktion wird neu gecalled####")
                # print()
                alignment = align_spans(
                    poss_g2p, poss_p2g, ops=["delO", "no-choice", "intrsct"]
                )
                self.documentation[id]["alignments"] = {}
                self.documentation[id]["alignments"]["alignment"] = alignment
                self.documentation[id]["alignments"]["jaccard"] = []
                self.documentation[id]["alignments"]["pred_tag"] = []
                self.documentation[id]["alignments"]["counts"] = []

                for gold_alignm in alignment:
                    gold_span = gld_spn[gold_alignm]
                    for pred_alignm in alignment[gold_alignm]:
                        pred_span = prd_spn[pred_alignm]

                        js = calc_jaccard_score(gold_span, pred_span, ib)

                        pred_tag = [pred_span[i] for i in pred_span][0]

                        self.documentation[id]["alignments"]["jaccard"].append(js)
                        self.documentation[id]["alignments"]["pred_tag"].append(
                            pred_tag
                        )

                        if js == 0:
                            if pred_tag == "O":
                                # FN
                                self.documentation[id]["alignments"]["counts"].append(
                                    "fn"
                                )
                                self.fn += 1
                            else:
                                # FP
                                self.documentation[id]["alignments"]["counts"].append(
                                    "fp"
                                )
                                self.fp += 1

                        elif js > 0 and js < threshold:
                            if pred_tag == "O":
                                # would have been TN if intersection would have been above threshold
                                self.documentation[id]["alignments"]["counts"].append(
                                    "-"
                                )
                                pass
                            else:
                                # would have been TP if intersection would have been above threshold
                                self.documentation[id]["alignments"]["counts"].append(
                                    "-"
                                )
                                pass
                        else:
                            if pred_tag == ".":
                                # full stops are not evaluated
                                self.documentation[id]["alignments"]["counts"].append(
                                    "-"
                                )
                                pass
                            elif pred_tag == "O":
                                # TN are not needed to calculate F1
                                self.documentation[id]["alignments"]["counts"].append(
                                    "tn(-)"
                                )
                                pass
                            else:
                                # TP
                                self.documentation[id]["alignments"]["counts"].append(
                                    "tp"
                                )
                                self.tp += 1

    def PrintDoc(self):
        for id in self.documentation:
            print(id)
            print(self.documentation[id])
            print()
