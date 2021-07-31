__author__ = "Maximilian Wegge"

from emotion import Data, Dataset, HMM, Evaluation

# from emotion import HMM
# from emotion import Evaluation


# read Data from file, only gne, all labels
emo_stim = Data(
    filename="data/rectified-unified-with-offsets.json",
    roles=["cause"],
    corpora=["emotion-stimulus"],
    splits=[0.8, 0.2],
)

# convert data to Brown-Format
emo_stim.conv2brown()

# set train- and test-sets
train = Dataset(data=emo_stim, splt=0)
test = Dataset(data=emo_stim, splt=1)

# train model on trainset, one for each label
hmm_emo_stim = HMM("cause")
hmm_emo_stim.train(dataset=train)

# predict testset using previously trained model
hmm_emo_stim.predictDataset(dataset=test)

# evaluate the predictions and return precicion, recall and f-score
eval_emo_stim = Evaluation(dataset=test, role="cause", threshold=0.8)

# eval_emo_stim.SaveDoc("tests/emo-stim/doc_emo_stim.json")

eval_emo_stim.save_eval(
    eval_name="stimulus", filename="tests/emo-stim/eval_emo_stim.json"
)
