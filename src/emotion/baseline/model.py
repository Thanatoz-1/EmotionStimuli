__author__ = "Tushar Dhyani"

import os
import pickle
from ..utils import Dataset, counter, logging

logger = logging.getLogger(__name__)


class HMM:
    def __init__(self, label=None) -> None:
        """
        Note that this implementation assumes that n, m, and T are small
        enough not to require underflow mitigation.

        Required Inputs:
        - transmission_prob: an (n+2) x (n+2) numpy array, initial, where n is
        the number of hidden states
        - emission_prob: an (m x n) 2-D numpy array, where m is the number of
        possible observations

        Optional Input:
        - obs: a list of observation labels, in the same order as their
        occurence within the emission probability matrix; otherwise, will assume
        that the emission probabilities are in alpha-numerical order.
        """
        self.label = label
        self.words_with_tags = list()
        self.word_given_tag_dict = dict()
        self.transitionMatrix = None
        self.uniqueTags = []

    def prob_word_given_tag(self, word, tag):
        """
        Estimates transmission probabilities from word to tag.
        """
        try:
            ans = self.word_given_tag_dict[word.lower(), tag] / self.tagCounter[tag]
        except:
            ans = 0.00000000001
        return ans

    def prob_tag2_given_tag1(self, tag2, tag1):
        """
        Count occurance of tag2 given tag1
        Take the counts and return probablity wrt tag1_counts. (tag2_given_tag1/tag1)
        """
        # Write assert cases
        t1_counts = 0
        t2_given_t1_counts = 0
        for i in range(len(self.words_with_tags) - 1):
            if self.words_with_tags[i][1] == tag1:
                t1_counts += 1
                if self.words_with_tags[i + 1][1] == tag2:
                    t2_given_t1_counts += 1
        return t2_given_t1_counts / (t1_counts + 1)

    def viterbi(self, sentence):
        """
        sentence should be same as the dataset, in the following form:
        List[List[tuple(str, str)]]
        example:
        [
            [
                (this, "O"),
                ("is", "B"),
                ("a", "O),
                ("sentence", "O")
            ]
        ]
        """
        # Write assert cases for non trained models
        state_prob = {}
        for tag in self.uniqueTags:
            for i in range(len(sentence)):
                state_prob[(tag, i)] = [
                    "prob of this state",
                    ["prev tag", "prev index"],
                ]
        for i, tup in enumerate(sentence):
            if i == 0:  # For the first run
                for tag in self.uniqueTags:
                    prob_transition = self.transitionMatrix[self.uniqueTags.index(".")][
                        self.uniqueTags.index(tag)
                    ]
                    tag_state_prob = (
                        self.prob_word_given_tag(tup[0], tag) * prob_transition
                    )
                    prevTag = "."

                    state_prob[(tag, i)][0] = tag_state_prob
                    state_prob[(tag, i)][1][0] = prevTag
            else:
                for tag_curr in self.uniqueTags:
                    tempTagState = []
                    for tag_prev in self.uniqueTags:
                        prob_transition = self.transitionMatrix[
                            self.uniqueTags.index(tag_prev)
                        ][self.uniqueTags.index(tag_curr)]
                        tag_state_prob = (
                            state_prob[(tag_prev, i - 1)][0]
                            * self.prob_word_given_tag(tup[0], tag)
                            * prob_transition
                        )
                        tempTagState.append(tag_state_prob)
                    maxTag = self.uniqueTags[tempTagState.index(max(tempTagState))]
                    # print(f"Line 23: MaxTag: {maxTag}, TagCurr: {tag_curr}")
                    prob_transition = self.transitionMatrix[
                        self.uniqueTags.index(maxTag)
                    ][self.uniqueTags.index(tag_curr)]
                    tag_state_prob = (
                        state_prob[(maxTag, i - 1)][0]
                        * self.prob_word_given_tag(tup[0], tag_curr)
                        * prob_transition
                    )
                    state_prob[(tag_curr, i)][0] = tag_state_prob
                    state_prob[(tag_curr, i)][1] = (maxTag, i - 1)

        FinalTagSequence = []
        Max = -1
        idx = len(sentence) - 1
        FinalState = ["tag", idx]
        for tag in self.uniqueTags:
            if state_prob[(tag, idx)][0] > Max:
                Max = state_prob[(tag, idx)][0]
                final_tag = tag
                FinalState = state_prob[(tag, idx)]
        tag_2_add = final_tag
        FinalTagSequence.append(tag_2_add)
        temp_tag = FinalState[1][0]
        temp_idx = idx - 1

        while temp_idx >= 0:
            FinalTagSequence.append(temp_tag)

            FinalState = state_prob[(temp_tag, temp_idx)]

            temp_tag = FinalState[1][0]
            temp_idx = temp_idx - 1
        FinalTagSequence.reverse()
        words = [word for word, tag in sentence]
        return list(zip(words, FinalTagSequence))

    def train(self, dataset: Dataset):
        """
        Dataset should be in the following form:
        List[List[tuple(str, str)]]
        example:
        [
            [
                (this, "O"),
                ("is", "B"),
                ("a", "O),
                ("sentence", "O"),
                ('.','.')
            ]
        ]

        Args:
            dataset: Dataset = The list of data in the format mentioned above.
        """
        for id in dataset.instances:
            if self.label in dataset.instances[id].roles:
                for tup in dataset.instances[id].gold[self.label]:
                    self.words_with_tags.append((tup[0].lower(), tup[1]))
            self.tagCounter = counter(tag for word, tag in self.words_with_tags)

        # Create the word given tag dictionary
        for id in dataset.instances:
            if self.label in dataset.instances[id].roles:
                for word, tag in dataset.instances[id].gold[self.label]:
                    try:
                        self.word_given_tag_dict[(word.lower(), tag)] += 1
                    except:
                        self.word_given_tag_dict[(word.lower(), tag)] = 1
        # Find unique tags and crate transition matrix from the same.
        self.uniqueTags = list({tup[1] for tup in self.word_given_tag_dict})
        self.transitionMatrix = [[0] * len(self.uniqueTags) for _ in self.uniqueTags]
        # Transition matrix = probablity of a tag given other tag
        for idx, tagi in enumerate(self.uniqueTags):
            for jdx, tagj in enumerate(self.uniqueTags):
                self.transitionMatrix[idx][jdx] = self.prob_tag2_given_tag1(tagj, tagi)

    def save(self, save_path: str = "~/.hmm"):
        """Function for saving the HMM model weights.

        Args:
            save_path (str): the path to the pkl file. Defaults to "~/.hmm".
        """
        with open(os.path.join(save_path, "tm.pkl"), "wb") as f:
            pickle.dump(self.transitionMatrix, f)

    def load(self, load_path: str = "~/.hmm"):
        """Function to load the pickle file from the path.

        Args:
            load_path (str): The path for the model weights (pkl file). Defaults to "~/.hmm".
        """
        with open(os.path.join(load_path, "tm.pkl"), "rb") as f:
            self.transitionMatrix = pickle.load(f)

    def predictSentence(self, sentence: list, verbose=False):
        tokens = [(i.lower(), "O") for i in sentence]
        prediction = self.viterbi(tokens)
        if verbose:
            prediction = [
                {"token": i[0], "gold": i[1], "pred": j[1]}
                for i, j in zip(tokens, prediction)
            ]
        return prediction

    def predictDataset(self, dataset: Dataset):
        for id in dataset.instances:
            # print(id)
            # gold = dataset.instances[id].gold[self.label]
            if self.label in dataset.instances[id].gold:
                to_predict = [(tok.lower(), "") for tok in dataset.instances[id].tokens]
                prediction = self.viterbi(to_predict)
                dataset.instances[id].pred[self.label] = prediction
        return None
