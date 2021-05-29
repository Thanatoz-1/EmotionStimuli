from emotion.utils import Data
from emotion.dataset import Dataset, Instance
from emotion import HMM
from emotion.evaluation import Evaluation


# read Data from file, only gne, all labels
eca = Data(
    filename="data/rectified-unified-with-offsets.json",
    labelset=["cause"],
    corpora=["eca"],
    splits=[0.8, 0.2],
)

# convert data to Brown-Format
eca.conv2brown()

# set train- and test-sets
train = Dataset(data=eca, splt=0)
test = Dataset(data=eca, splt=1)

# train model on trainset, one for each label
hmm_eca = HMM("cause")
hmm_eca.train(dataset=train)

# predict testset using previously trained model
hmm_eca.predictDataset(dataset=test)

# evaluate the predictions and return precicion, recall and f-score
eval_eca = Evaluation(dataset=test, label="cause", threshold=0.8)

# eval_eca.SaveDoc("doc_eca.json")

eval_eca.SaveEval("eca.txt")
