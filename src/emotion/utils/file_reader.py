import random, json, copy


class Data:
    def __init__(
        self,
        filename: str,
        labelset: list = ["cause", "cue", "experiencer", "target"],
        corpora: list = ["eca", "emotion-stimulus", "reman", "gne"],
        splits: list = [1],
    ):
        self.data = []
        self.splits = splits
        self.split_data = []

        self.ReadFile(filename, labelset, corpora)
        self.SplitData()

    def ReadFile(self, filename: str, labelset, corpora):
        self.data.clear()
        with open(filename, "r") as file:
            all_data = json.load(file)

        for instance in all_data:
            if instance["dataset"] in corpora:
                relevant = copy.deepcopy(instance)
                for label in instance["annotations"]:
                    if label not in labelset:
                        relevant["annotations"].pop(label)
                self.data.append(relevant)

    def SplitData(self):
        self.split_data.clear()
        cpy_data = copy.deepcopy(self.data)
        random.seed(10)
        random.shuffle(cpy_data)
        not_split = copy.deepcopy(cpy_data)
        for splt in self.splits:
            splt_point = int(splt * len(cpy_data))
            self.split_data.append(not_split[:splt_point])
            not_split = not_split[splt_point:]

    def conv2brown(self):
        for instance in self.data:
            tokens = instance["tokens"]
            orig = instance["annotations"]
            brown = {}
            for label in orig:
                brown[label] = []
                for tup in zip(tokens, orig[label]):
                    if tup[0] == ".":
                        brown[label].append((tup[0], "."))
                    else:
                        brown[label].append((tup[0].lower(), tup[1]))
            instance["annotations"] = brown

        self.SplitData()
