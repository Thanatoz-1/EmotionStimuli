import random, json

class File:
    def __init__(self):
        self.data = []
        self.train = []
        self.dev = []
        self.test = []
    
    def read(self, filename):
        with open(filename,'r') as file:
            self.data = json.load(file)
    
    def splitData(self, train=0.8, dev=0.1, test=0.1):
        if train + dev + test == 1:
            random.seed(10)
            random.shuffle(self.data)
            train_split = int(train*len(self.data))
            test_split = int((1-test)*len(self.data))
            self.train = self.data[:train_split]
            self.dev = self.data[train_split:test_split]
            self.test = self.data[test_split:]
        else:
            pass

    def getData(self):
        return self.data

    def getTrain(self):
        return self.train
    
    def getDev(self):
        return self.dev

    def getTest(self):
        return self.test
