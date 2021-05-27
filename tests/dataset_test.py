from emotion.utils import Data #, simPred
from emotion.dataset import Dataset
from emotion import HMM

gne = Data(filename='data/rectified-unified-with-offsets.json', corpora = ['gne'])
gne.SplitData(splits=[0.9,0.08,0.02])

train = Dataset(data=gne, splt=1)
test = Dataset(data=gne,splt=2)
#test = Dataset(data=gne)

#simPred(test)
hmm = HMM('experiencer', train, test)
hmm.train()
#print(hmm.transitionMatrix)
#print(hmm.uniqueTags)

#hmm.predictDataset()
test.predict(model=hmm)
#print(hmm.testing.ReturnInst()[-1].pred_annots)
#print(len(test.ReturnInst()))



#test.EvalCounts()
#test.EvalMetrics()

#print(test.ReturnMetrics())
