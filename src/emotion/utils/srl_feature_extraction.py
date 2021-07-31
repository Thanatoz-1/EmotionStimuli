__author__ = "Maximilian Wegge"

# This is a stand-alone file which we ran once; it is not part of our framework.

# from .practnlptools import Annotator
# To run this program, practNLPTools has to be installed. If it is not,
# the installation of our framework will fail, as the import cannot be executed.
# Therefore, it is ignored here.
from transformers import BertTokenizer
import spacy
import json

# we rely on the BERT tokenizer to join the tokenized sentences (as the SR-Labeler)
# only accepts non-tokenized input) and to re-tokenize the SR-Labeler's output.

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased", lower=False)


def gen_input(dataset: dict):
    """Join each of the tokenized sentences in the data to one string.

    Args:
        dataset (dict): input data.

    Returns:
        srl_input (dict): all previously tokenized sentences in our data
        as strings, ordered by ids.
    """
    srl_input = {}
    for inst in dataset:
        idx = inst["id"]
        toks = inst["tokens"]
        encodes = []
        for tok in toks:
            encodes += tokenizer.encode_plus(tok, add_special_tokens=False).input_ids
        joined = tokenizer.decode(encodes)
        srl_input[idx] = joined
    return srl_input


def getsrl(srl_input: dict):
    """Pass all sentences of our data to practNLPTools in order to generate the SRL.

    Args:
        srl_input (dict): all sentences (not tokenized) in our data, ordered by their ids
            (e.g., reman-0).

    Returns:
        errors (set): potentially occuring errors in generating the SRL output (for debugging).
        srl_output (dict): all SRL-outputs per sentence, ordered by their ids.
    """
    annot = Annotator()
    srl_output = {}
    errors = set([])
    for idx in srl_input:
        sent = srl_input[idx]
        # as a good SRL-output is strongly dependent on the correct punctuation of
        # the input, a period is added if not already in place (will be omitted in the
        # output, thus not changing the number of tokens)
        if sent[-1] != ".":
            sent += "."
        try:
            srl_analysis = annot.getAnnotations(sent)["srl"]
            srl_output[idx] = srl_analysis
        except:
            errors.add(idx)

    return errors, srl_output


def getsrl_encoding(srl_output: dict):
    """generate dictionaries for encoding the SRL as SRL-ids and vice-versa.

    Args:
        srl_output (dict): all SRL-outputs per sentence, ordered by their ids.

    Returns:
        srl2id (dict), id2srl (dict): dictionaries for encoding the SRL as SRL-ids (srl2id)
            and vice-versa (id2srl).
    """
    srl2id = {}
    id2srl = {}

    # get a set of all SR-labels in the SRL-output
    all_srl = []
    for k in srl_output:
        for outp in srl_output[k]:
            all_srl.extend(list(outp.keys()))
    all_srl = set(all_srl)

    # dict for encoding SRL as SRL-id
    srl2id["X"] = 0  # X denotes the tokens without any corresponding SRL
    for role, index in zip(all_srl, range(len(all_srl))):
        srl2id[role] = index + 1

    # dict for encoding SRL-id as SRL
    for role in srl2id:
        index = srl2id[role]
        id2srl[index] = role

    return srl2id, id2srl


def align_sent(srl_sent: list, tokens: dict, srl2id: dict):
    """map the SRL output of one sentence to the tokens in the
    original (tokenized) input sentence.

    Args:
        srl_sent (list(dict)): SRL output for one sentence. Might include multiple subsets
            of SRL if more than one verb if the input sentence has more than one verb.
        tokens (dict): all tokenized sentences in our data, ordered by their ids.
        srl2id (dict): dictionary for encoding the SRL as SRL-ids.

    Returns:
        aligned (list(list)): all SRL for the input sentence, mapped to the original tokenized
            sentence. Multiple SRL-outpus (one for each verb) stored in sub-lists respectively.
    """
    aligned = []

    # iterate over each subset of the SRL-output of the sentence
    # (when multiple words in input sentence)
    for subset in srl_sent:
        subs_srl = []
        subs_aligned = []

        # tokenized the output and map each token of the output to its corresponding SRL
        for srl in subset:
            subs_srl.extend([(tok, srl) for tok in [i.text for i in nlp(subset[srl])]])

        # map the SRL to the token-sequence of the original sentence
        # encode the SRL by its SRL-id
        for tok in tokens:
            if subs_srl != [] and tok == subs_srl[0][0]:
                subs_aligned.append(srl2id[subs_srl[0][1]])
                subs_srl.pop(0)
            else:
                subs_aligned.append(0)

        # collect the (encoded) SRL for all subsets in one list for the overall sentence
        aligned.append(subs_aligned)

    return aligned


def align_all(srl_output: dict, toks: dict, srl2id: dict):
    """map all SRL outputs to the tokens in their respective (tokenized) input sentence.

    Args:
        srl_output (dict): all SRL-outputs per sentence, ordered by their ids.
        toks (dict): original (tokenized) input sentences, ordered by their ids.
        srl2id (dict): dictionary for encoding the SRL as SRL-ids.

    Returns:
        alignments (dict): all tokens in all sentences mapped to their respective SRL.
            (multiple subsets per sentence if multiple verbs in sentence).
    """
    srl_alignments = {}
    for idx in srl_output:
        srl_sent = srl_output[idx]
        tokens = toks[idx]
        srl_alignments[idx] = align_sent(srl_sent, tokens, srl2id)

    return srl_alignments


def save_srl_features_all(srl_output: dict, all_tokens: dict, srl2id: dict):
    """align SRL features and save as json.

    Args:
        srl_output (dict): all SRL-outputs per sentence, ordered by their ids.
        all_tokens (dict): all orignal, tokenized sentences.
        srl2id (dict): dictionary for encoding the SRL as SRL-ids.

    Returns:
        srl_features_all (dict): all SRL features aligned with their corresponding input sentence (encoded with SRL-ids).
    """

    srl_features_all = align_all(srl_output, all_tokens, srl2id)

    with open("data/srl_features_all.json", "w", encoding="utf-8") as outp:
        outp.write(json.dumps(srl_features_all))

    return srl_features_all


def save_srl_features_sctl(er: str, srl_features_all: dict, srl2id: dict):
    """generate specific SRL features for one emotion role and save as json.

    Args:
        er (str): emotion role for which the specific featureset is generated.
        srl_features_all (dict): all tokens in all sentences mapped to their respective SRL.
            (multiple subsets per sentence if multiple verbs in sentence).
        srl2id (dict): dictionary for encoding the SRL as SRL-ids

    Returns:
        None
    """
    selected_srl = {}
    # list of all SRL relevant for the specific emotion role
    all_relevant_srl = {
        "exp": [srl2id["A0"], srl2id["A1"]],
        "targ": [srl2id["A0"], srl2id["A1"], srl2id["A2"], srl2id["V"]],
        "cue": [
            srl2id["A0"],
            srl2id["A1"],
            srl2id["A2"],
            srl2id["V"],
        ],
        "cse": [
            srl2id["A0"],
            srl2id["A1"],
            srl2id["A2"],
            srl2id["A3"],
            srl2id["A4"],
            srl2id["V"],
            srl2id["AM-TMP"],
            srl2id["AM-ADV"],
            srl2id["AM-LOC"],
            srl2id["AM-PNC"],
            srl2id["AM-MNR"],
            srl2id["AM-MOD"],
            srl2id["AM-CAU"],
        ],
    }
    relevant_srl = all_relevant_srl[er]

    for k in srl_features_all:
        selected_srl[k] = []

        for outp in srl_features_all[k]:
            tmp = []
            before_verb = True
            for i in outp:
                if i == srl2id["V"]:
                    before_verb = False
                if er == "exp" and i == srl2id["A1"] and before_verb:
                    tmp.append(i)
                elif er == "tar" and i == srl2id["A0"] and not before_verb:
                    tmp.append(i)
                elif i in relevant_srl:
                    tmp.append(i)
                else:
                    tmp.append(0)
            selected_srl[k].append(tmp)

    with open("data/srl_features_" + er + ".json", "w", encoding="utf-8") as outp:
        outp.write(json.dumps(selected_srl))

    return None


def main():
    """run functions to generate and save all SRL featuresets.

    Returns:
        None
    """

    dataset = json.load(open("data/rectified-unified-with-offset.json", "r"))

    srl_input = gen_input(dataset)

    srl_output = getsrl(srl_input)[1]

    srl2id, id2srl = getsrl_encoding(srl_output)

    # list of all orignal, tokenized sentences
    all_tokens = {}
    for inst in dataset:
        all_tokens[inst["id"]] = inst["tokens"]

    srl_features_all = save_srl_features_all(srl_output, all_tokens, srl2id)

    for er in ["exp", "targ", "cue", "cse"]:
        save_srl_features_sctl(er, srl_features_all, srl2id)

    return None
