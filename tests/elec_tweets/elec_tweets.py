from emotion.utils import Data
from emotion.dataset import Dataset, Instance
from emotion import HMM
from emotion.evaluation import Evaluation


# read Data from file, only gne, all labels
elec_all = Data(
    filename="data/rectified-unified-with-offsets.json",
    labelset=["experiencer", "target", "cue", "cause"],
    corpora=["electoral_tweets"],
    splits=[0.8, 0.2],
)

# convert data to Brown-Format
elec_all.conv2brown()

# set train- and test-sets
train = Dataset(data=elec_all, splt=0)
test = Dataset(data=elec_all, splt=1)

# train model on trainset, one for each label
hmm_elec_exp = HMM("experiencer")
hmm_elec_tar = HMM("target")
hmm_elec_cue = HMM("cue")
hmm_elec_cse = HMM("cause")

hmm_elec_exp.train(dataset=train)
hmm_elec_tar.train(dataset=train)
hmm_elec_cue.train(dataset=train)
hmm_elec_cse.train(dataset=train)

# predict testset using previously trained model
hmm_elec_exp.predictDataset(dataset=test)
hmm_elec_tar.predictDataset(dataset=test)
hmm_elec_cue.predictDataset(dataset=test)
hmm_elec_cse.predictDataset(dataset=test)

# evaluate the predictions and return precicion, recall and f-score
eval_elec_exp = Evaluation(dataset=test, label="experiencer", threshold=0.8)
eval_elec_tar = Evaluation(dataset=test, label="target", threshold=0.8)
eval_elec_cue = Evaluation(dataset=test, label="cue", threshold=0.8)
eval_elec_cse = Evaluation(dataset=test, label="cause", threshold=0.8)

eval_elec_exp.SaveDoc("tests/elec_tweets/doc_elec_exp.json")
# eval_elec_tar.SaveDoc("tests/elec_tweets/doc_elec_tar.json")
# eval_elec_cue.SaveDoc("tests/elec_tweets/doc_elec_cue.json")
# eval_elec_cse.SaveDoc("tests/elec_tweets/doc_elec_cse.json")

eval_elec_exp.SaveEval("tests/elec_tweets/eval_elec_exp.txt")
eval_elec_tar.SaveEval("tests/elec_tweets/eval_elec_tar.txt")
eval_elec_cue.SaveEval("tests/elec_tweets/eval_elec_cue.txt")
eval_elec_cse.SaveEval("tests/elec_tweets/eval_elec_cse.txt")
