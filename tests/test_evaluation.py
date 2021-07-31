__author__ = "Maximilian Wegge"
from emotion import Data, Dataset, Instance, HMM, Evaluation

gne_exp = Data(
    filename="data/rectified-unified-with-offsets.json",
    allow_roles=["experiencer"],
    allow_corpora=["reman"],
    splits=[0],
)

gne_exp.conv2brown()

test = Dataset(data=gne_exp, splt=0)

testtokens = ["Das", "ist", "ein", "Test", "das", "ist", "ein", "Test"]
testannot = ["B", "O", "O", "B", "I", "I", "I", "I"]
testpredict = ["B", "I", "I", "B", "I", "I", "I", "I"]

test.instances["testid"] = Instance(
    tokens=testtokens,
    corpus="testcorpus",
)
test.instances["testid"].set_gld(
    role="experiencer",
    annotation=[tup for tup in zip(testtokens, testannot)],
)
# test.instances["testid"].InitPred(label="experiencer")

test.instances["testid"].pred["experiencer"] = [
    tup for tup in zip(testtokens, testpredict)
]

eval_gne2gne_exp = Evaluation(dataset=test, role="experiencer", threshold=0.8)

print(eval_gne2gne_exp.precision)
print(eval_gne2gne_exp.recall)
print(eval_gne2gne_exp.fscore)

eval_gne2gne_exp.save_doc(filename="tests/test_doc.json")
eval_gne2gne_exp.save_eval(eval_name="gne2gne_exp", filename="tests/test_eval.json")
