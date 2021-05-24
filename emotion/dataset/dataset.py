from ..evaluation import align_spans, gen_poss_align, jaccard_score, calc_precision, calc_recall, calc_fscore
from ..utils import Data
import random

class Dataset:

    def __init__(self, data:Data, splt=0):
        self.instances = []
        self.labels = []
        #self.eval = {'prc':0, 'rec':0, 'f1':0}
        self.eval = {}
        self.counts = {}
        self.loadData(data, splt)

        
    def loadData(self, data:Data, splt=0):
        instances = []
        data = data.getSplit(splt)
        for i in range(len(data)):
            instance = data[i]
            dataset = instance['dataset']
            tokens = instance['tokens']
            labels = instance['annotations']
            self.labels = set(self.labels).union(set(labels))
            id = instance['id']
            span = Sentence(id=id,tokens=tokens)
            null = len(tokens)*['O']

            if dataset == 'eca' or dataset == 'emotion-stimulus':
                if 'cause' not in labels:
                    span.setGold(role='cause',annotation=null)
                else:
                    span.setGold(role='cause',annotation=labels['cause'])
            
            elif dataset == 'reman':
                if 'experiencer' not in labels:
                    span.setGold(role='experiencer',annotation=null)
                else:
                    span.setGold(role='experiencer',annotation=labels['experiencer'])
                
                if 'target' not in labels:
                    span.setGold(role='target',annotation=null)
                else:
                    span.setGold(role='target',annotation=labels['target'])

                if 'cause' not in labels:
                    span.setGold(role='cause',annotation=null)
                else:
                    span.setGold(role='cause',annotation=labels['cause'])
                    
                if 'cue' not in labels:
                    span.setGold(role='cue',annotation=null)
                else:
                    span.setGold(role='cue',annotation=labels['cue'])

            else:
                if 'experiencer' not in labels:
                    span.setGold(role='experiencer',annotation=null)
                else:
                    span.setGold(role='experiencer',annotation=labels['experiencer'])
                
                if 'target' not in labels:
                    span.setGold(role='target',annotation=null)
                else:
                    span.setGold(role='target',annotation=labels['target'])

                if 'cause' not in labels:
                    span.setGold(role='cause',annotation=null)
                else:
                    span.setGold(role='cause',annotation=labels['cause'])
                    
                if 'cue' not in labels:
                    span.setGold(role='cue',annotation=null)
                else:
                    span.setGold(role='cue',annotation=labels['cue'])

            instances.append(span)

        for annotation in self.labels:
            self.counts[annotation] = {'tp':0, 'fp':0, 'fn':0}
            self.eval[annotation] = {'prc':0, 'rec':0, 'f1':0}
        self.instances = instances


    def getInst(self):
        return self.instances
    

    def getCounts(self, threshold):
        for sentence in self.instances:
            sentence.eval_counts(threshold)
            sen_counts = sentence.getCounts()
            for annotation in sen_counts:
                self.counts[annotation]['tp'] += sen_counts[annotation]['tp']
                self.counts[annotation]['fp'] += sen_counts[annotation]['fp']
                self.counts[annotation]['fn'] += sen_counts[annotation]['fn']


    def evaluate(self, threshold=0.8):
        self.getCounts(threshold)
        for annotation in self.counts:
            tp = self.counts[annotation]['tp']
            fp = self.counts[annotation]['fp']
            fn = self.counts[annotation]['fn']
            self.eval[annotation]['prc'] = calc_precision(tp, fp)
            self.eval[annotation]['rec'] = calc_recall(tp, fn)
            precision = self.eval[annotation]['prc']
            recall = self.eval[annotation]['rec']
            self.eval[annotation]['f1'] = calc_fscore(precision, recall)
    

    def getEval(self):
        return self.eval

class Sentence:

    def __init__(self, id:str, tokens:list):
        self.id = id
        self.tokens = tokens
        self.gld_tags = {}
        self.prd_tags = {}
        self.labels = []
        self.gld_spans = {} #gldspans = {Exp:[{},{}], Tar:[{0:'O',1:'O',2:'O'},{3:'B', 4:'I', 5:'I'},{6:'O', 7:'O'}],}
        self.prd_spans = {} #prdspans = {Exp:[{},{}], Tar:[{0:'O',1:'O'}, {2:'B',3:'I', 4:'I', 5:'I'},{6:'O', 7:'O'}],}
        self.counts = {}


    def setGold(self, role:str, annotation:list):
        self.gld_tags[role] = annotation
        self.gld_spans[role] = []
        tmp = {}
        for i in range (len(annotation)):
            iob = annotation[i]
            if i == 0:
                tmp[i] = iob
            elif iob == 'I' and tmp[i-1] in ['B', 'I']:
                tmp[i] = iob
            elif iob == 'O' and tmp[i-1] == 'O':
                tmp[i] = iob
            else:
                self.gld_spans[role].append(tmp)
                tmp = {}
                tmp[i] = iob
            
            if i == len(annotation)-1:
                self.gld_spans[role].append(tmp)
            else:
                pass
        
        self.labels = [x for x in self.gld_tags.keys()]
        for annotation in self.labels:
            self.counts[annotation] = {'tp':0, 'fp': 0, 'fn': 0}


    def setPred(self, role:str, annotation:list):
        self.prd_tags[role] = annotation
        self.prd_spans[role] = []
        tmp = {}
        for i in range(len(annotation)):
            iob = annotation[i]
            if i == 0:
                tmp[i] = iob
            elif iob == 'I' and tmp[i-1] in ['B', 'I']:
                tmp[i] = iob
            elif iob == 'O' and tmp[i-1] == 'O':
                tmp[i] = iob
            else:
                self.prd_spans[role].append(tmp)
                tmp = {}
                tmp[i] = iob
            
            if i == len(annotation)-1:
                self.prd_spans[role].append(tmp)
            else:
                pass


    def eval_counts(self, threshold):
        for annotation in self.gld_tags:
            gold = self.gld_spans[annotation]
            pred = self.prd_spans[annotation]
            poss_g2p = gen_poss_align(frm=gold, to=pred)
            poss_p2g = gen_poss_align(frm=pred, to=gold)
            alignment = align_spans(poss_g2p, poss_p2g)
            for g_spn in alignment:
                #gld_span = list(gold[alignment].values())
                gld_span = gold[g_spn]
                for p_spn in alignment[g_spn]:
                    #prd_span = list(pred[span].values())
                    prd_span = pred[p_spn]
                    js = jaccard_score(gld_span, prd_span)
                    idx = list(prd_span)[0]
                    pred_tag = prd_span[idx]
                    if js == 0:
                        if pred_tag == 'O':
                            self.counts[annotation]['fn'] += 1 # FN
                        else:
                            self.counts[annotation]['fp'] += 1 # FP
                    elif js > 0 and js < threshold:
                        pass
                    else:
                        if pred_tag == 'O':
                            pass # TN are not needed
                            #self.counts['tn'] += 1 # TN
                        else:
                            self.counts[annotation]['tp'] += 1 # TP


    def getTokens(self):
        return self.tokens


    def getId(self):
        return self.id


    def getLabels(self):
        return self.labels


    def getCounts(self):
        return self.counts


    def getGoldSpan(self, role='all'):
        if role == 'all':
            return self.gld_spans
        else:
            return self.gld_spans[role]


    def getPrdSpan(self, role='all'):
        if role == 'all':
            return self.prd_spans
        else:
            return self.prd_spans[role]


    def getGldTag(self, role='all'):
        if role == 'all':
            return self.gld_tags
        else:
            return self.gld_tags[role]
    

    def getPrdTag(self, role='all'):
        if role == 'all':
            return self.prd_tags
        else:
            return self.prd_tags[role]