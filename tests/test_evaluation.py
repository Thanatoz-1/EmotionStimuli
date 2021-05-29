from emotion.utils import Data
from emotion.dataset import Dataset, Instance
from emotion import HMM
from emotion.evaluation import Evaluation

gne_exp = Data(
    filename="data/rectified-unified-with-offsets.json",
    labelset=["experiencer"],
    corpora=["gne"],
    splits=[0],
)

gne_exp.conv2brown()

test = Dataset(data=gne_exp, splt=0)

testtokens = ["Das", "ist", "ein", "Test", "das", "ist", "ein", "Test"]
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
test.instances["testid"].InitPred(label="experiencer")

test.instances["testid"].pred["experiencer"] = [
    tup for tup in zip(testtokens, testpredcit)
]

eval_gne2gne_exp = Evaluation(dataset=test, threshold=0.8)

# print(eval_gne2gne_exp.precision)
# print(eval_gne2gne_exp.recall)
# print(eval_gne2gne_exp.fscore)

eval_gne2gne_exp.PrintDoc()
