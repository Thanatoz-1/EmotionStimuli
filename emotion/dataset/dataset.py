from ..evaluation import jaccard_score

class Dataset:

    def __init__(self, data, whitelist=['eca', 'emotion-stimulus', 'reman', 'gne']):
        self.whitelist = whitelist
        self.instances = self.loadData(data)
        
        
    def loadData(self, data):
        instances = []
        for i in range(len(data)):
            instance = data[i]
            dataset = instance['dataset']

            if dataset in self.whitelist:
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
            else:
                pass

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
        self.g_annotation = {}
        self.p_annotation = {}
        self.goldspans = {}
        self.predspans = {}
        self.jaccard = {}

    # set attributes
    def setGold(self, role:str, annotation:list):
        self.g_annotation[role] = annotation
        self.goldspans[role] = []
        tmp = {}
        for i in range (len(annotation)):
            iob = annotation[i]
            if iob != 'O':
                tmp[i] = iob
                if i == len(annotation)-1:
                    self.goldspans[role].append(tmp)
            elif tmp != {}:
                self.goldspans[role].append(tmp)
                tmp = {}
            else:
                pass

    def setPred(self, role:str, annotation:list):
        self.p_annotation[role] = annotation
        self.predspans[role] = []
        tmp = {}
        for i in range(len(annotation)):
            iob = annotation[i]
            if iob != 'O':
                tmp[i] = iob
                if i == len(annotation)-1:
                    self.predspans[role].append(tmp)
            elif tmp != {}:
                self.predspans[role].append(tmp)
                tmp = {}
            else:
                pass

    def calcJaccard(self):
        for annotation in self.g_annotation:
            gold = self.getGold(annotation)
            pred = self.getPred(annotation)[:]
            for gold_span in gold:
                for pred_span in pred:
                    score = jaccard(gold, pred)
            self.jaccard[annotation] = score

    # get attributes
    def getTokens(self):
        return self.tokens

    def getId(self):
        return self.id

    def getLabelset(self):
        return [x for x in self.g_annotation.keys()]

    def getGold(self, role='all'):
        if role == 'all':
            return self.goldspans
        else:
            return self.goldspans[role]

    def getPred(self, role='all'):
        if role == 'all':
            return self.predspans
        else:
            return self.predspans[role]

    def getGAnnotation(self, role='all'):
        if role == 'all':
            return self.g_annotation
        else:
            return self.g_annotation[role]
    
    def getPAnnotation(self, role='all'):
        if role == 'all':
            return self.p_annotation
        else:
            return self.p_annotation[role]