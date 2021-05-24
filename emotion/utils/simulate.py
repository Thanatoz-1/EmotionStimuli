from emotion.dataset.dataset import Dataset
import random

def simPred(dataset:Dataset):
    for inst in dataset.getInst():
        for label in inst.getLabels():
            tokens = inst.getTokens()
            tags = 60*['O'] + 15*['B'] + 25*['I']
            random.shuffle(tags)
            prediction = []
            for i in range(len(tokens)):
                prediction.append(tags[random.randint(0,len(tags)-1)])
            inst.setPred(role=label,annotation=prediction)