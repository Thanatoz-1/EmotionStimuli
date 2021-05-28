from emotion.utils import Data
from emotion.dataset import Dataset, Instance
from emotion import HMM
from emotion.evaluation import Evaluation


def gettags(ls):
    return [tup[1] for tup in ls]


gne_exp = Data(
    filename="data/rectified-unified-with-offsets.json",
    labelset=["experiencer"],
    corpora=["gne"],
    splits=[0.8, 0.2],
)

gne_exp.conv2brown()

train = Dataset(data=gne_exp, splt=0)
test = Dataset(data=gne_exp, splt=1)

################################################################################
"""testtokens = ["Das", "ist", "ein", "Test", "das", "ist", "ein", "Test"]
testannot = ["B", "O", "O", "B", "I", "I", "I", "I"]
testpredcit = ["B", "O", "O", "O", "O", "B", "I", "I"]

test.instances["testid"] = Instance(
    tokens=testtokens,
    corpus="testcorpus",
)
test.instances["testid"].SetGold(
    label="experiencer",
    annotation=[tup for tup in zip(testtokens, testannot)],
)
test.instances["testid"].InitPred(label="experiencer")"""
################################################################################

hmm_exp = HMM("experiencer")
hmm_exp.train(train)
# print(hmm_exp.transitionMatrix)
# print(hmm_exp.uniqueTags)

hmm_exp.predictDataset(dataset=test)

################################################################################
"""test.instances["testid"].pred["experiencer"] = [
    tup for tup in zip(testtokens, testpredcit)
]"""
################################################################################

eval_gne2gne_exp = Evaluation(dataset=test, threshold=0.8)
print(eval_gne2gne_exp.precision)
print(eval_gne2gne_exp.recall)
print(eval_gne2gne_exp.fscore)

# eval_gne2gne_exp.PrintDoc()
