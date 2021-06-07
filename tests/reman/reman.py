from emotion.utils import Data, Dataset
from emotion import HMM
from emotion.evaluation import Evaluation


# read Data from file, only gne, all labels
rem_all = Data(
    filename="data/rectified-unified-with-offsets.json",
    roles=[
        "experiencer",
        "target",
        "cue-joy",
        "cue-fear",
        "cue-trust",
        "cue-surprise",
        "cue-anticipation",
        "cue-sadness",
        "cue-anger",
        "cue-disgust" "cue-other",
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
hmm_rem_cue_joy = HMM("cue-joy")
hmm_rem_cue_fear = HMM("cue-fear")
hmm_rem_cue_trust = HMM("cue-trust")
hmm_rem_cue_surpr = HMM("cue-surprise")
hmm_rem_cue_sadn = HMM("cue-sadness")
hmm_rem_cue_anger = HMM("cue-anger")
hmm_rem_cue_antic = HMM("cue-anticipation")
hmm_rem_cue_disg = HMM("cue-disgust")
hmm_rem_cue_other = HMM("cue-other")
hmm_rem_cse = HMM("cause")

hmm_rem_exp.train(dataset=train)
hmm_rem_tar.train(dataset=train)
hmm_rem_cue_joy.train(dataset=train)
hmm_rem_cue_fear.train(dataset=train)
hmm_rem_cue_trust.train(dataset=train)
hmm_rem_cue_surpr.train(dataset=train)
hmm_rem_cue_sadn.train(dataset=train)
hmm_rem_cue_anger.train(dataset=train)
hmm_rem_cue_antic.train(dataset=train)
hmm_rem_cue_disg.train(dataset=train)
hmm_rem_cue_other.train(dataset=train)
hmm_rem_cse.train(dataset=train)

# predict testset using previously trained model
hmm_rem_exp.predictDataset(dataset=test)
hmm_rem_tar.predictDataset(dataset=test)
hmm_rem_cue_joy.predictDataset(dataset=test)
hmm_rem_cue_fear.predictDataset(dataset=test)
hmm_rem_cue_trust.predictDataset(dataset=test)
hmm_rem_cue_surpr.predictDataset(dataset=test)
hmm_rem_cue_sadn.predictDataset(dataset=test)
hmm_rem_cue_anger.predictDataset(dataset=test)
hmm_rem_cue_antic.predictDataset(dataset=test)
hmm_rem_cue_other.predictDataset(dataset=test)
hmm_rem_cue_disg.predictDataset(dataset=test)
hmm_rem_cse.predictDataset(dataset=test)

# evaluate the predictions and return precicion, recall and f-score
eval_rem_exp = Evaluation(dataset=test, role="experiencer", threshold=0.8)
eval_rem_tar = Evaluation(dataset=test, role="target", threshold=0.8)
eval_rem_cue_joy = Evaluation(dataset=test, role="cue-joy", threshold=0.8)
eval_rem_cue_fear = Evaluation(dataset=test, role="cue-fear", threshold=0.8)
eval_rem_cue_trust = Evaluation(dataset=test, role="cue-trust", threshold=0.8)
eval_rem_cue_surpr = Evaluation(dataset=test, role="cue-surprise", threshold=0.8)
eval_rem_cue_sadn = Evaluation(dataset=test, role="cue-sadness", threshold=0.8)
eval_rem_cue_anger = Evaluation(dataset=test, role="cue-anger", threshold=0.8)
eval_rem_cue_antic = Evaluation(dataset=test, role="cue-anticipation", threshold=0.8)
eval_rem_cue_disg = Evaluation(dataset=test, role="cue-disgust", threshold=0.8)
eval_rem_cue_other = Evaluation(dataset=test, role="cue-other", threshold=0.8)
eval_rem_cse = Evaluation(dataset=test, role="cause", threshold=0.8)

# eval_rem_exp.save_doc("tests/reman/doc_reman_exp.json")
# eval_rem_tar.save_doc("tests/reman/doc_reman_tar.json")
# eval_rem_cue_joy.SaveDoc("tests/reman/doc_reman_cue_joy.json")
# eval_rem_cue_fear.SaveDoc("tests/reman/doc_reman_cue_fear.json")
# eval_rem_cue_trust.SaveDoc("tests/reman/doc_reman_cue_trust.json")
# eval_rem_cue_surpr.SaveDoc("tests/reman/doc_reman_cue_surpr.json")
# eval_rem_cue_sadn.SaveDoc("tests/reman/doc_reman_cue_sadn.json")
# eval_rem_cue_anger.SaveDoc("tests/reman/doc_reman_cue_anger.json")
# eval_rem_cue_antic.SaveDoc("tests/reman/doc_reman_cue_antic.json")
# eval_rem_cue_disg.SaveDoc("tests/reman/doc_reman_cue_disg.json")
# eval_rem_cue_other.SaveDoc("tests/reman/doc_reman_cue_other.json")
# eval_rem_cse.SaveDoc("tests/reman/doc_reman_cse.json")

eval_rem_exp.save_eval(eval_name="experiencer", filename="tests/reman/eval_reman.json")
eval_rem_tar.save_eval(eval_name="target", filename="tests/reman/eval_reman.json")
eval_rem_cue_joy.save_eval(eval_name="cue_joy", filename="tests/reman/eval_reman.json")
eval_rem_cue_fear.save_eval(
    eval_name="cue_fear", filename="tests/reman/eval_reman.json"
)
eval_rem_cue_trust.save_eval(
    eval_name="cue_trust", filename="tests/reman/eval_reman.json"
)
eval_rem_cue_surpr.save_eval(
    eval_name="cue_surprise", filename="tests/reman/eval_reman.json"
)
eval_rem_cue_sadn.save_eval(
    eval_name="cue_sadness", filename="tests/reman/eval_reman.json"
)
eval_rem_cue_anger.save_eval(
    eval_name="cue_anger", filename="tests/reman/eval_reman.json"
)
eval_rem_cue_antic.save_eval(
    eval_name="cue_anticipation", filename="tests/reman/eval_reman.json"
)
eval_rem_cue_disg.save_eval(
    eval_name="cue_disgust", filename="tests/reman/eval_reman.json"
)
eval_rem_cue_other.save_eval(
    eval_name="cue_other", filename="tests/reman/eval_reman.json"
)
eval_rem_cse.save_eval(eval_name="cause", filename="tests/reman/eval_reman.json")
