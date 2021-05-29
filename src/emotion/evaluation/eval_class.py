from os import write
from .metrics import calc_precision, calc_recall, calc_fscore, calc_jaccard_score
from .utils import gen_poss_align, align_spans
from ..utils.utils import conv2span
from ..dataset import Dataset
import json


class Evaluation:
    def __init__(
        self, dataset: Dataset, label: str, threshold: int = 0.8, beta=1, ib=True
    ) -> None:
        self.label = label
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.documentation = {}
        self.evaluate(dataset=dataset, ib=ib)
        self.precision = calc_precision(tp=self.tp, fp=self.fp)
        self.recall = calc_recall(tp=self.tp, fn=self.fn)
        self.fscore = calc_fscore(prec=self.precision, rec=self.recall, beta=beta)

    def evaluate(self, dataset, ib):
        for id in dataset.instances:

            gld_spn = conv2span(dataset.instances[id].gold[self.label])
            prd_spn = conv2span(dataset.instances[id].pred[self.label])

            poss_g2p = gen_poss_align(frm=gld_spn, to=prd_spn)
            poss_p2g = gen_poss_align(frm=prd_spn, to=gld_spn)

            alignment = align_spans(
                poss_g2p, poss_p2g, ops=["delO", "no-choice", "intrsct"]
            )

            self.documentation[id] = {}
            self.documentation[id]["tokens"] = dataset.instances[id].tokens
            self.documentation[id]["spans"] = {}
            self.documentation[id]["spans"]["gold"] = gld_spn
            self.documentation[id]["annotations"] = {}
            self.documentation[id]["spans"]["gold"] = prd_spn
            self.documentation[id]["eval"] = {}
            self.documentation[id]["eval"]["jaccard"] = []
            self.documentation[id]["eval"]["pred_tag"] = []
            self.documentation[id]["eval"]["counts"] = []
            self.documentation[id]["annotations"]["gold"] = dataset.instances[id].gold[
                self.label
            ]
            self.documentation[id]["annotations"]["pred"] = dataset.instances[id].pred[
                self.label
            ]
            self.documentation[id]["eval"]["alignment"] = alignment

            for gold_alignm in alignment:
                gold_span = gld_spn[gold_alignm]
                for pred_alignm in alignment[gold_alignm]:
                    pred_span = prd_spn[pred_alignm]

                    js = calc_jaccard_score(gold_span, pred_span, ib)

                    pred_tag = [pred_span[i] for i in pred_span][0]

                    self.documentation[id]["eval"]["jaccard"].append(js)
                    self.documentation[id]["eval"]["pred_tag"].append(pred_tag)

                    if js == 0:
                        if pred_tag == "O":
                            # FN
                            self.documentation[id]["eval"]["counts"].append("fn")
                            self.fn += 1
                        else:
                            # FP
                            self.documentation[id]["eval"]["counts"].append("fp")
                            self.fp += 1

                    elif js > 0 and js < self.threshold:
                        if pred_tag == "O":
                            # would have been TN if intersection would have been above threshold
                            self.documentation[id]["eval"]["counts"].append("-")
                            pass
                        else:
                            # would have been TP if intersection would have been above threshold
                            self.documentation[id]["eval"]["counts"].append("-")
                            pass
                    else:
                        if pred_tag == ".":
                            # full stops are not evaluated
                            self.documentation[id]["eval"]["counts"].append("-")
                            pass
                        elif pred_tag == "O":
                            # TN are not needed to calculate F1
                            self.documentation[id]["eval"]["counts"].append("tn(-)")
                            pass
                        else:
                            # TP
                            self.documentation[id]["eval"]["counts"].append("tp")
                            self.tp += 1

    def PrintDoc(self):
        for id in self.documentation:
            print(id)
            print(self.documentation[id])
            print()

    def SaveDoc(self, filename: str) -> None:
        with open(filename, "w") as file:
            json.dump(self.documentation, file)

    def SaveEval(self, filename: str) -> None:
        with open(filename, "w") as file:
            file.write(f"size:\t\t\t\t{len(self.documentation)}\n")
            file.write(f"evaluated label:\t{self.label}\n")
            file.write(f"threshold:\t\t\t{self.threshold}\n")
            file.write(f"amount of tp:\t\t{self.tp}\n")
            file.write(f"amount of fn:\t\t{self.fn}\n")
            file.write(f"amount of fp:\t\t{self.fp}\n")
            file.write(f"precision:\t\t\t{self.precision}\n")
            file.write(f"recall:\t\t\t\t{self.recall}\n")
            file.write(f"f-score:\t\t\t{self.fscore}")
