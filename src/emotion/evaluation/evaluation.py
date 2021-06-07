__author__ = "Maximilian Wegge"
from emotion import evaluation
from .metrics import calc_precision, calc_recall, calc_fscore
from .align_spans import gen_poss_align, align_spans
from .convert_to_span import conv2span
from ..utils import Dataset
import json


class Evaluation:
    """Stores the results and documentation of the evaluation."""

    def __init__(
        self, dataset: Dataset, role: str, threshold: int = 0.8, beta=1.0
    ) -> None:
        """Initialize the Evaluation object.

        Args:
            dataset (Dataset): Dataset object containing the gold
            as well as the predicted annotations.
            annotations.
            role (str): the emotion role for which the annotations should be
            evaluated.
            threshold (int, optional): A predicted span of tags is only evaluated
            to TP if the jaccard score of this span and the aligned gold span
            is above the threshold. Defaults to 0.8.
            beta (int, optional): Value of beta used for calculating f-score.
            Defaults to 1.
        """
        self.role = role
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.documentation = {}
        self.beta = beta
        self.evaluate(dataset=dataset)
        self.precision = calc_precision(tp=self.tp, fp=self.fp)
        self.recall = calc_recall(tp=self.tp, fn=self.fn)
        self.fscore = calc_fscore(prec=self.precision, rec=self.recall, beta=self.beta)

    def evaluate(self, dataset) -> None:
        """accumulate TP, FP and FN of all instances in the dataset
        and store the details of this evaluation.

        Args:
            dataset ([type]): Dataset object containing the gold
            as well as the predicted annotations.
        """
        for id in dataset.instances:

            if self.role in dataset.instances[id].gold:
                # get the spans of tags in the gold and the predicted annotations.
                gld_spn = conv2span(dataset.instances[id].gold[self.role])
                prd_spn = conv2span(dataset.instances[id].pred[self.role])

                # generate all possible alignment__g2ps between the spans.
                poss_g2p = gen_poss_align(frm=gld_spn, to=prd_spn)
                poss_p2g = gen_poss_align(frm=prd_spn, to=gld_spn)

                # delete ambiguous alignment__g2ps to obtain final alignment__g2p
                alignment_g2p = align_spans(
                    poss_g2p,
                    poss_p2g,
                    ops=["del_O", "del_no_choice", "del_shortest_jaccard", "del_rand"],
                )

                self.documentation[id] = {}
                self.documentation[id]["tokens"] = dataset.instances[id].tokens
                self.documentation[id]["spans"] = {}
                self.documentation[id]["spans"]["gold"] = gld_spn
                self.documentation[id]["annotations"] = {}
                self.documentation[id]["spans"]["pred"] = prd_spn
                self.documentation[id]["eval"] = {}
                self.documentation[id]["eval"]["jaccard"] = []
                self.documentation[id]["eval"]["pred_tag"] = []
                self.documentation[id]["eval"]["counts"] = []
                self.documentation[id]["annotations"]["gold"] = dataset.instances[
                    id
                ].gold[self.role]
                self.documentation[id]["annotations"]["pred"] = dataset.instances[
                    id
                ].pred[self.role]
                self.documentation[id]["eval"]["alignment"] = str(alignment_g2p)

                for gold_alignm in alignment_g2p:

                    for pred_span in alignment_g2p[gold_alignm]:

                        pred_alignm = alignment_g2p[gold_alignm][pred_span]

                        jaccard = pred_alignm[1]
                        pred_tag = pred_alignm[0]

                        self.documentation[id]["eval"]["jaccard"].append(jaccard)
                        self.documentation[id]["eval"]["pred_tag"].append(pred_tag)

                        if jaccard == 0:
                            if pred_tag == "O":
                                # evaluates to FN.
                                self.documentation[id]["eval"]["counts"].append("fn")
                                self.fn += 1
                            else:
                                # evaluates to FP.
                                self.documentation[id]["eval"]["counts"].append("fp")
                                self.fp += 1

                        elif jaccard > 0 and jaccard < self.threshold:
                            if pred_tag == "O":
                                # would evaluate to TN if jaccard-score would have
                                # been above threshold.
                                self.documentation[id]["eval"]["counts"].append(
                                    "- (tn<thrshld)"
                                )
                                pass
                            else:
                                # would evalaute to TP if jaccard-score would have
                                # been above threshold. Thus, evaluates to FN.
                                self.documentation[id]["eval"]["counts"].append(
                                    "fn (<- tp<thrshld)"
                                )
                                self.fn += 1
                                pass
                        else:
                            if pred_tag == ".":
                                # full stops are not evaluated.
                                self.documentation[id]["eval"]["counts"].append("-")
                                pass
                            elif pred_tag == "O":
                                # evaluates to TN but is omitted here as TN are not
                                # relevant for calculating precision, recall and f-score.
                                self.documentation[id]["eval"]["counts"].append(
                                    "- (tn)"
                                )
                                pass
                            else:
                                # evaluates to TP.
                                self.documentation[id]["eval"]["counts"].append("tp")
                                self.tp += 1

    def print_doc(self) -> None:
        """Print the documentation."""
        for id in self.documentation:
            print(f"ID:\t{id}\n{self.documentation[id]}\n")
        return None

    def print_eval(self) -> None:
        """Print calulated Precision, Recall and F-Score."""
        print(
            f"precision:\t{self.precision}\nrecall:\t{self.recall[id]}\nf-score:\t{self.fscore}"
        )
        return None

    def save_doc(self, filename: str) -> None:
        """Save the documentation in a new json-file.

        Args:
            filename (str): name and path of the json-file.
        """
        with open(filename, "w") as doc_file:
            json.dump(self.documentation, doc_file)

    def save_eval(self, eval_name: str, filename: str) -> None:
        """Save the documentation either in an existing or a new json-file.

        Args:
            eval_name (str): name of the evaluation. Needed to identify the
            evaluation if one file contains multiple evaluations.
            filename (str): name of either an existing json-file containing
            evaluations or the name of a new file.
        """
        try:
            with open(filename, "r") as eval_file:
                data = eval_file.read()
                eval_output = json.loads(data)
                eval_output[eval_name] = {}

        except:
            eval_output = {eval_name: {}}

        eval_output[eval_name]["size"] = len(self.documentation)
        eval_output[eval_name]["evaluated role"] = self.role
        eval_output[eval_name]["jaccard-threshold"] = self.threshold
        eval_output[eval_name]["amount TP"] = self.tp
        eval_output[eval_name]["amount FN"] = self.fn
        eval_output[eval_name]["amount FP"] = self.fp
        eval_output[eval_name]["precision"] = self.precision
        eval_output[eval_name]["recall"] = self.recall
        eval_output[eval_name]["f-score"] = self.fscore
        eval_output[eval_name]["beta for f-score"] = self.beta

        with open(filename, "w") as eval_file:
            json.dump(eval_output, eval_file)

        return None
