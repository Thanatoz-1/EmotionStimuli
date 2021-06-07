from emotion.dataset.dataset import Dataset
import random

def simPred(dataset:Dataset):
    counter = 23
    for inst in dataset.ReturnInst():
        for label in inst.ReturnLabels():
            tokens = inst.ReturnTokens()
            tags = 60*['O'] + 15*['B'] + 25*['I']
            random.seed(counter)
            random.shuffle(tags)
            counter += 1
            prediction = []
            for i in range(len(tokens)):
                prediction.append(tags[random.randint(0,len(tags)-1)])
            inst.SetPred(label=label,annotation=prediction)