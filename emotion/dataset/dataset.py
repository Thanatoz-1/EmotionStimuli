from ..evaluation import jaccard_score
from ..utils import Data
import random

class Dataset:

    def __init__(self, data:Data):
        self.instances = self.loadData(data)
        
    def loadData(self, data:Data, splt=0):
        instances = []
        data = data.getSplit(splt)
        for i in range(len(data)):
            instance = data[i]
            dataset = instance['dataset']
            tokens = instance['tokens']
            labels = instance['annotations']
            id = instance['id']
            span = Sentence(id=id,tokens=tokens)
            null = len(tokens)*['O']

            if dataset == 'eca' or dataset == 'emotion-stimulus':
                if 'cause' not in labels:
                    span.setGold(role='Sti',annotation=null)
                else:
                    span.setGold(role='Sti',annotation=labels['cause'])
            
            elif dataset == 'reman':
                if 'experiencer' not in labels:
                    span.setGold(role='Exp',annotation=null)
                else:
                    span.setGold(role='Exp',annotation=labels['experiencer'])
                
                if 'target' not in labels:
                    span.setGold(role='Tar',annotation=null)
                else:
                    span.setGold(role='Tar',annotation=labels['target'])

                if 'cause' not in labels:
                    span.setGold(role='Sti',annotation=null)
                else:
                    span.setGold(role='Sti',annotation=labels['cause'])
                    
                if 'cue' not in labels:
                    span.setGold(role='Cue',annotation=null)
                else:
                    span.setGold(role='Cue',annotation=labels['cue'])

            else:
                if 'experiencer' not in labels:
                    span.setGold(role='Exp',annotation=null)
                else:
                    span.setGold(role='Exp',annotation=labels['experiencer'])
                
                if 'target' not in labels:
                    span.setGold(role='Tar',annotation=null)
                else:
                    span.setGold(role='Tar',annotation=labels['target'])

                if 'cause' not in labels:
                    span.setGold(role='Sti',annotation=null)
                else:
                    span.setGold(role='Sti',annotation=labels['cause'])
                    
                if 'cue' not in labels:
                    span.setGold(role='Cue',annotation=null)
                else:
                    span.setGold(role='Cue',annotation=labels['cue'])

            instances.append(span)

        return instances

    def getInst(self):
        return self.instances
    
    def evaluate(self):
        pass

class Sentence:
    # initialize class
    def __init__(self, id:str, tokens:list):
        self.id = id
        self.tokens = tokens
        self.gld_tags = {}
        self.prd_tags = {}
        self.gld_spans = {}
        self.prd_spans = {}
        self.jaccard = {}

    # set attributes
    def setGold(self, role:str, annotation:list):
        self.gld_tags[role] = annotation
        self.gld_spans[role] = []
        tmp = {}
        for i in range (len(annotation)):
            iob = annotation[i]
            if iob != 'O':
                tmp[i] = iob
                if i == len(annotation)-1:
                    self.gld_spans[role].append(tmp)
            elif tmp != {}:
                self.gld_spans[role].append(tmp)
                tmp = {}
            else:
                pass

    def setPred(self, role:str, annotation:list):
        self.prd_tags[role] = annotation
        self.prd_spans[role] = []
        tmp = {}
        for i in range(len(annotation)):
            iob = annotation[i]
            if iob != 'O':
                tmp[i] = iob
                if i == len(annotation)-1:
                    self.prd_spans[role].append(tmp)
            elif tmp != {}:
                self.prd_spans[role].append(tmp)
                tmp = {}
            else:
                pass

    def calcJcc(self):
        for annotation in self.gld_tags:
            gold = self.getGldSpans(annotation)
            pred = self.getPrdSpans(annotation)[:]
            for gld_span in gold:
                tmp = []
                for prd_span in pred:
                    score = jaccard_score(gld_span, prd_span)
                    tmp.append(score)
                max(tmp)
            self.jaccard[annotation] = score

    # get attributes
    def getTokens(self):
        return self.tokens

    def getId(self):
        return self.id

    def getLabels(self):
        return [x for x in self.gld_tags.keys()]

    def getGoldSpan(self, role='all'):
        if role == 'all':
            return self.gld_spans
        else:
            return self.gld_spans[role]

    def getPrdSpan(self, role='all'):
        if role == 'all':
            return self.prd_spans
        else:
            return self.predspans[role]

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