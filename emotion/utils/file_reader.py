import random, json

class Data:
    def __init__(self, filename:str, corpora=['eca', 'emotion-stimulus', 'reman', 'gne']):
        self.data = []
        self.splits = []
        self.splitinfo = ['data not split']
        self.corpusinfo = ['no corpora loaded']
        self.loadFromFile(filename, corpora)

    def loadFromFile(self, filename:str, corpora=['eca', 'emotion-stimulus', 'reman', 'gne']):
        self.corpusinfo = corpora
        with open(filename,'r') as file:
            raw_data = json.load(file)
        data = []
        for instance in raw_data:
            if instance['dataset'] in corpora:
                data.append(instance)            
        self.data = data
        self.splits.append(data)

    def splitData(self, splits=[0.8,0.1,0.1]):
        if sum(splits) != 1:
            print ('Splits must sum up to 1')
            return None
        self.splitinfo = splits
        self.splits = []
        random.seed(10)
        random.shuffle(self.data)
        raw = self.data[:]
        for splt in splits:
            splt_point = int(splt*len(self.data))
            self.splits.append(raw[:splt_point])
            raw = raw[splt_point:] 
    
    def getInfo(self, info_on:str):
        if info_on == 'splits':
            return self.splitinfo
        elif info_on == 'corpus':
            return self.corpusinfo
        else:
            pass

    def getSplit(self, splt:int):
        return self.splits[splt]

    def getData(self):
        return self.data
