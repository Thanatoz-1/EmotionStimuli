from emotion.utils import Data
from emotion.dataset import Dataset, Instance
from emotion import HMM
from emotion.evaluation import Evaluation


# read Data from file, only gne, all labels
rem_all = Data(
    filename="data/rectified-unified-with-offsets.json",
    labelset=[
        "experiencer",
        "target",
        "cue-joy",
        "cue-sadness",
        "cue-anger",
        "cue-other",
        "cause",
    ],
    corpora=["reman"],
    splits=[0.8, 0.2],
)

# convert data to Brown-Format
rem_all.conv2brown()

# set train- and test-sets
train = Dataset(data=rem_all, splt=0)
test = Dataset(data=rem_all, splt=1)

# train model on trainset, one for each label
hmm_rem_exp = HMM("experiencer")
hmm_rem_tar = HMM("target")
hmm_rem_cue_j = HMM("cue-joy")
hmm_rem_cue_a = HMM("cue-anger")
hmm_rem_cue_s = HMM("cue-sadness")
hmm_rem_cue_o = HMM("cue-other")
hmm_rem_cse = HMM("cause")

hmm_rem_exp.train(dataset=train)
hmm_rem_tar.train(dataset=train)
hmm_rem_cue_j.train(dataset=train)
hmm_rem_cue_a.train(dataset=train)
hmm_rem_cue_s.train(dataset=train)
hmm_rem_cue_o.train(dataset=train)
hmm_rem_cse.train(dataset=train)

# predict testset using previously trained model
hmm_rem_exp.predictDataset(dataset=test)
hmm_rem_tar.predictDataset(dataset=test)
hmm_rem_cue_j.predictDataset(dataset=test)
hmm_rem_cue_a.predictDataset(dataset=test)
hmm_rem_cue_s.predictDataset(dataset=test)
hmm_rem_cue_o.predictDataset(dataset=test)
hmm_rem_cse.predictDataset(dataset=test)

# evaluate the predictions and return precicion, recall and f-score
eval_gne_exp = Evaluation(dataset=test, label="experiencer", threshold=0.8)
eval_gne_tar = Evaluation(dataset=test, label="target", threshold=0.8)
eval_gne_cue_j = Evaluation(dataset=test, label="cue-joy", threshold=0.8)
eval_gne_cue_a = Evaluation(dataset=test, label="cue-anger", threshold=0.8)
eval_gne_cue_s = Evaluation(dataset=test, label="cue-sadness", threshold=0.8)
eval_gne_cue_o = Evaluation(dataset=test, label="cue-other", threshold=0.8)
eval_gne_cse = Evaluation(dataset=test, label="cause", threshold=0.8)

# eval_gne_exp.SaveDoc("tests/reman/doc_gne_exp.json")
# eval_gne_tar.SaveDoc("tests/reman/doc_gne_tar.txt")
# eval_gne_cue_j.SaveDoc("tests/reman/doc_gne_cue_j.txt")
# eval_gne_cue_a.SaveDoc("tests/reman/doc_gne_cue_a.txt")
# eval_gne_cue_s.SaveDoc("tests/reman/doc_gne_cue_s.txt")
# eval_gne_cue_o.SaveDoc("tests/reman/doc_gne_cue_o.txt")
# eval_gne_cse.SaveDoc("tests/reman/doc_gne_cse.txt")

eval_gne_exp.SaveEval("tests/reman/eval_gne_exp.txt")
eval_gne_tar.SaveEval("tests/reman/eval_gne_tar.txt")
eval_gne_cue_j.SaveEval("tests/reman/eval_gne_cue_j.txt")
eval_gne_cue_a.SaveEval("tests/reman/eval_gne_cue_a.txt")
eval_gne_cue_s.SaveEval("tests/reman/eval_gne_cue_s.txt")
eval_gne_cue_o.SaveEval("tests/reman/eval_gne_cue_o.txt")
eval_gne_cse.SaveEval("tests/reman/eval_gne_cse.txt")
