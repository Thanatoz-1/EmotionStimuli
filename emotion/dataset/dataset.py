from ..evaluation import align_spans, gen_poss_align, jaccard_score, calc_precision, calc_recall, calc_fscore
from ..utils import Data
import random

def conv2span(annotation):
    span = []
    tmp = {}
    for i in range (len(annotation)):
        tag = annotation[i] # IOB tag
        if i == 0:
            tmp[i] = tag
        elif tag == 'I' and tmp[i-1] in ['B', 'I']:
            tmp[i] = tag
        elif tag == 'O' and tmp[i-1] == 'O':
            tmp[i] = tag
        else:
            span.append(tmp)
            tmp = {}
            tmp[i] = tag
        
        if i == len(annotation)-1:
            span.append(tmp)
        else:
            pass
    return span


class Dataset:

    def __init__(self, data:Data, splt=0):
        self.instances = []
        self.labels = set()
        self.counts = {}
        self.metrics = {}
        self.LoadData(data, splt)

        
    def LoadData(self, data:Data, splt=0):
        loaded_inst = []
        source = data.ReturnSplit(splt)
        for i in range(len(source)):
            src_inst = source[i]
            src_corpus = src_inst['dataset']
            src_tokens = src_inst['tokens']
            src_annotations = src_inst['annotations']
            src_id = src_inst['id']
            instance = Instance(id=src_id,tokens=src_tokens)
            #null = len(tokens)*['O']

            for label in src_annotations:
                self.labels.add(label)
                instance.SetGold(label=label, annotation=src_annotations[label])

            loaded_inst.append(instance)

        self.instances = loaded_inst
        self.ResetEval()
 

    def EvalCounts(self, threshold:int=0.8):
        for sentence in self.instances:
            sentence.EvalCounts(threshold)
            sen_counts = sentence.ReturnCounts()
            for annotation in sen_counts:
                self.counts[annotation]['tp'] += sen_counts[annotation]['tp']
                self.counts[annotation]['fp'] += sen_counts[annotation]['fp']
                self.counts[annotation]['fn'] += sen_counts[annotation]['fn']


    def EvalMetrics(self):
        for annotation in self.counts:
            tp = self.counts[annotation]['tp']
            fp = self.counts[annotation]['fp']
            fn = self.counts[annotation]['fn']
            self.metrics[annotation]['prc'] = calc_precision(tp, fp)
            self.metrics[annotation]['rec'] = calc_recall(tp, fn)
            precision = self.metrics[annotation]['prc']
            recall = self.metrics[annotation]['rec']
            self.metrics[annotation]['f1'] = calc_fscore(precision, recall)
    

    def ResetEval(self):
        for label in self.labels:
            self.counts[label] = {'tp':0, 'fp':0, 'fn':0}
            self.metrics[label] = {'prc':0, 'rec':0, 'f1':0}


    def ReturnMetrics(self):
        return self.metrics


    def ReturnInst(self):
        return self.instances


class Instance:

    def __init__(self, id:str, tokens:list):
        self.id = id
        self.tokens = tokens
        self.gold_annots = {}
        self.gold_spans = {} # = {Exp:[{},{}], Tar:[{0:'O',1:'O',2:'O'},{3:'B', 4:'I', 5:'I'},{6:'O', 7:'O'}],}
        self.pred_annots = {}
        self.pred_spans = {} # = {Exp:[{},{}], Tar:[{0:'O',1:'O'}, {2:'B',3:'I', 4:'I', 5:'I'},{6:'O', 7:'O'}],}
        self.labels = set()
        self.counts = {}


    def SetGold(self, label:str, annotation:list):
        self.gold_annots[label] = annotation
        span = conv2span(annotation)
        self.gold_spans[label] = span
        
        self.labels.add(label)


    def SetPred(self, label:str, annotation:list):
        self.pred_annots[label] = annotation
        span = conv2span(annotation)
        self.pred_spans[label] =span


    def ResetCounts(self):
        for label in self.labels:
            self.counts[label] = {'tp':0, 'fp': 0, 'fn': 0}


    def EvalCounts(self, threshold):
        self.ResetCounts()
        for label in self.labels:
            all_gld_spns = self.gold_spans[label]
            all_prd_spns = self.pred_spans[label]
            poss_g2p = gen_poss_align(frm=all_gld_spns, to=all_prd_spns)
            poss_p2g = gen_poss_align(frm=all_prd_spns, to=all_gld_spns)
            alignment = align_spans(poss_g2p, poss_p2g)
            for gold_alignm in alignment:
                gold_span = all_gld_spns[gold_alignm]
                for pred_alignm in alignment[gold_alignm]:
                    pred_span = all_prd_spns[pred_alignm]

                    js = jaccard_score(gold_span, pred_span)

                    pred_tag = [pred_span[i] for i in pred_span][0]
                    
                    if js == 0:
                        if pred_tag == 'O':
                            self.counts[label]['fn'] += 1 # FN
                        else:
                            self.counts[label]['fp'] += 1 # FP
                    elif js > 0 and js < threshold:
                        pass
                    else:
                        if pred_tag == 'O':
                            pass # TN are not needed
                            #self.counts['tn'] += 1 # TN
                        else:
                            self.counts[label]['tp'] += 1 # TP


    def ReturnTokens(self):
        return self.tokens


    def ReturnId(self):
        return self.id


    def ReturnLabels(self):
        return self.labels


    def ReturnCounts(self):
        return self.counts


    def ReturnGoldSpans(self, label='all'):
        if label == 'all':
            return self.gold_spans
        else:
            return self.gold_spans[label]


    def ReturnPredSpans(self, label='all'):
        if label == 'all':
            return self.pred_spans
        else:
            return self.pred_spans[label]


    def ReturnGoldAnnots(self, label='all'):
        if label == 'all':
            return self.gold_annots
        else:
            return self.gold_annots[label]
    

    def ReturnPredAnnots(self, label='all'):
        if label == 'all':
            return self.pred_annots
        else:
            return self.pred_annots[label]