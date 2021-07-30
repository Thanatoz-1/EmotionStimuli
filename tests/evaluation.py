__author__ = "Maximilian Wegge"

import json
from emotion import HMM


def display_result(prediction):
    for i in prediction:
        if i.get("gold") == i.get("pred"):
            print(
                f'{i.get("token").ljust(15)}{i.get("gold").ljust(1)} {i.get("pred").ljust(3)}'
            )
        else:
            print(
                f'{i.get("token").ljust(15)}\x1b[0;37;41m{i.get("gold").ljust(1)}\x1b[0m {i.get("pred").ljust(3)}'
            )
    print()


data = [
    {
        "dataset": "reman",
        "id": "reman-3",
        "emotions": ["noemo"],
        "text": "The household had never been disturbed by all the going and coming .",
        "tokens": [
            "The",
            "household",
            "had",
            "never",
            "been",
            "disturbed",
            "by",
            "all",
            "the",
            "going",
            "and",
            "coming",
            ".",
        ],
        "annotations": {
            "experiencer": [
                "B",
                "I",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
            ],
            "cause": ["O", "O", "O", "O", "O", "O", "O", "B", "I", "I", "I", "I", "I"],
        },
        "meta": {"domain": "literature", "annotation": "expert"},
        "steps": ["extract", "tokenize", "split"],
        "tags": [],
        "split": "test",
        "annotation-offsets": {"experiencer": [[0, 13]], "cause": [[42, 68]]},
    }
]

data = json.load(open("data/rectified-unified-with-offsets.json", "r"))
# Hyper-params
targetted_label = "cause"
targetted_dataset = "reman"

# Convert to brown dataset format
dataset = []
for i in data:
    if i.get("dataset") == targetted_dataset:
        sub = []
        try:
            for tok, lab in zip(
                i.get("tokens"), i.get("annotations").get(targetted_label)
            ):
                if tok == ".":
                    sub.append((tok.lower(), "."))
                else:
                    sub.append((tok.lower(), lab))
            dataset.append(sub)
        except:
            continue
# Train
hmm = HMM(targetted_label)
hmm.train(dataset)
pred1 = hmm.predict(
    "the couple landed the helicopter in the middle of the forest and infuriated the authority",
    1,
)
display_result(pred1)
pred2 = hmm.predict(dataset[3], 1)
display_result(pred2)
# Make first predictions

# Done!
