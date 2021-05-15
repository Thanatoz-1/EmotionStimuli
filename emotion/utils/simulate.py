import random

def simPred(dataset):
    for inst in dataset.getInst():
        for label in inst.getLabelset():
            tokens = inst.getTokens()
            tags = 7*['O'] + 1*['B'] + 2*['I']
            random.shuffle(tags)
            prediction = []
            for i in range(len(tokens)):
                prediction.append(tags[random.randint(0,len(tags)-1)])
            inst.setPred(role=label,annotation=prediction)