import random, json

class Data:
    
    def __init__(self, filename:str, corpora=['eca', 'emotion-stimulus', 'reman', 'gne']):
        self.data = []
        self.splits = []
        self.ReadFile(filename, corpora)

    def ReadFile(self, filename:str, corpora=['eca', 'emotion-stimulus', 'reman', 'gne']):
        with open(filename,'r') as file:
            raw_data = json.load(file)
        data = []
        for instance in raw_data:
            if instance['dataset'] in corpora:
                data.append(instance)            
        self.data = data
        self.splits.append(data)

    def SplitData(self, splits=[0.8,0.1,0.1]):
        if sum(splits) != 1:
            print ('Splits must sum up to 1')
            return None

        self.splits = []

        random.seed(10)
        random.shuffle(self.data)
        unsplit = self.data[:]
        for splt in splits:
            splt_point = int(splt*len(self.data))
            self.splits.append(unsplit[:splt_point])
            unsplit = unsplit[splt_point:]
    
    def ReturnSplit(self, splt:int=0):
        return self.splits[splt]

    def ReturnData(self):
        return self.data
