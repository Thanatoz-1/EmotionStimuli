__author__ = "Maximilian Wegge"

from emotion.utils import Data, Dataset
from emotion import HMM
from emotion.evaluation import Evaluation


# read Data from file, only gne, all labels
elec_all = Data(
    filename="data/rectified-unified-with-offsets.json",
    roles=["experiencer", "target", "cue", "cause"],
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
eval_elec_exp = Evaluation(dataset=test, role="experiencer", threshold=0.8)
eval_elec_tar = Evaluation(dataset=test, role="target", threshold=0.8)
eval_elec_cue = Evaluation(dataset=test, role="cue", threshold=0.8)
eval_elec_cse = Evaluation(dataset=test, role="cause", threshold=0.8)

# eval_elec_exp.save_doc("tests/elec_tweets/doc_elec_exp.json")
# eval_elec_tar.save_doc("tests/elec_tweets/doc_elec_tar.json")
# eval_elec_cue.save_doc("tests/elec_tweets/doc_elec_cue.json")
# eval_elec_cse.save_doc("tests/elec_tweets/doc_elec_cse.json")

eval_elec_exp.save_eval(eval_name="exp", filename="tests/elec_tweets/eval_elec.json")
eval_elec_tar.save_eval(eval_name="target", filename="tests/elec_tweets/eval_elec.json")
eval_elec_cue.save_eval(eval_name="cue", filename="tests/elec_tweets/eval_elec.json")
eval_elec_cse.save_eval(eval_name="cause", filename="tests/elec_tweets/eval_elec.json")
