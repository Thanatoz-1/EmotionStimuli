from emotion.utils import Data
from emotion.dataset import Dataset, Instance
from emotion import HMM
from emotion.evaluation import Evaluation


# read Data from file, only gne, all labels
gne_all = Data(
    filename="data/rectified-unified-with-offsets.json",
    labelset=["experiencer", "target", "cue", "cause"],
    corpora=["gne"],
    splits=[0.8, 0.2],
)

# convert data to Brown-Format
gne_all.conv2brown()

# set train- and test-sets
train = Dataset(data=gne_all, splt=0)
test = Dataset(data=gne_all, splt=1)

# train model on trainset, one for each label
hmm_gne_exp = HMM("experiencer")
hmm_gne_tar = HMM("target")
hmm_gne_cue = HMM("cue")
hmm_gne_cse = HMM("cause")

hmm_gne_exp.train(dataset=train)
hmm_gne_tar.train(dataset=train)
hmm_gne_cue.train(dataset=train)
hmm_gne_cse.train(dataset=train)

# predict testset using previously trained model
hmm_gne_exp.predictDataset(dataset=test)
hmm_gne_tar.predictDataset(dataset=test)
hmm_gne_cue.predictDataset(dataset=test)
hmm_gne_cse.predictDataset(dataset=test)

# evaluate the predictions and return precicion, recall and f-score
eval_gne_exp = Evaluation(dataset=test, label="experiencer", threshold=0.8)
eval_gne_tar = Evaluation(dataset=test, label="target", threshold=0.8)
eval_gne_cue = Evaluation(dataset=test, label="cue", threshold=0.8)
eval_gne_cse = Evaluation(dataset=test, label="cause", threshold=0.8)

# eval_gne_exp.SaveDoc("doc_gne_exp.json")
# eval_gne_tar.SaveDoc("doc_gne_tar.txt")
# eval_gne_cue.SaveDoc("doc_gne_cue.txt")
# eval_gne_cse.SaveDoc("doc_gne_cse.txt")

eval_gne_exp.SaveEval("tests/gne/eval_gne_exp.txt")
eval_gne_tar.SaveEval("tests/gne/eval_gne_tar.txt")
eval_gne_cue.SaveEval("tests/gne/eval_gne_cue.txt")
eval_gne_cse.SaveEval("tests/gne/eval_gne_cse.txt")
