__author__ = "Maximilian Wegge"
from .file_reading import Data


class Dataset:
    def __init__(self, data: Data, splt: int = 0):
        self.instances = {}  # {id0: inst0, id1: inst1}
        self.labels = set()
        self.corpora = set()

        self.LoadData(data, splt)

    def LoadData(self, data: Data, splt: int = 0):
        source = data.split_data[splt]
        for i in range(len(source)):
            src_inst = source[i]
            src_corpus = src_inst["dataset"]
            src_tokens = src_inst["tokens"]
            src_annotations = src_inst["annotations"]
            src_id = src_inst["id"]

            instance = Instance(tokens=src_tokens, corpus=src_corpus)
            for label in src_annotations:
                self.labels.add(label)
                self.corpora.add(src_corpus)
                instance.SetGold(label=label, annotation=src_annotations[label])
                # instance.InitPred(label=label)
                self.instances[src_id] = instance


class Instance:
    def __init__(self, tokens: list, corpus: str):
        self.tokens = tokens  # ['the', 'household', 'will', 'never']
        self.gold = {}
        self.pred = {}
        self.labels = set()
        self.corpus = corpus

    def SetGold(self, label: str, annotation: list):
        self.labels.add(label)
        self.gold[label] = annotation

        # span = conv2span(annotation)
        # self.gold_spans[label] = span

    def ReturnTokens(self):
        return self.tokens

    def ReturnLabels(self):
        return self.labels

    def ReturnGoldAnnots(self, label="all"):
        if label == "all":
            return self.gold_annots
        else:
            return self.gold_annots[label]

    def ReturnPredAnnots(self, label="all"):
        if label == "all":
            return self.pred_annots
        else:
            return self.pred_annots[label]
