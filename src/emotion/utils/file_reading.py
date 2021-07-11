__author__ = "Maximilian Wegge"

import random, json, copy


class Data:
    """The Data object stores the unaltered data from a file
    and performs preprocessing steps such as splitting into
    subsets and converting the annotations' format.
    """

    def __init__(
        self,
        filename: str,
        roles: list = ["cause", "cue", "experiencer", "target"],
        corpora: list = ["eca", "emotion-stimulus", "reman", "gne"],
        splits: list = [1],
    ) -> None:
        """Initialize the Data object. Read data from file and split
        it into subsets.

        Args:
            filename (str): name of file containing the data.
            roles (list, optional): Specifies which emotion roles
            to read from the file. Defaults to ["cause", "cue", "experiencer", "target"].
            corpora (list, optional): Specifies which corpus/corpora
            to load from the file. Defaults to ["eca", "emotion-stimulus", "reman", "gne"].
            splits (list, optional): Specifies the size of subsets the data is split into.
            Defaults to [1].
        """
        self.data = []
        self.splits = splits  # metadata: amount/size of subsets.
        self.split_data = []
        self.ReadFile(filename, roles, corpora)
        self.SplitData()

    def ReadFile(self, filename: str, allow_roles: list, allow_corpora: list) -> None:
        """Load relevant data from file and store it in Data object.

        Args:
            filename (str): name of file containing the data.
            allow_roles (list): Specifies which emotion roles to read from the file.
            If there are no annotations for the given emotion roles, annotations of
            only 'O' are created.
            allow_corpora ([type]): Specifies which corpus/corpora to load from file.
        """
        self.data.clear()
        with open(filename, "r") as file:
            all_data = json.load(file)
        for instance in all_data:

            if instance["dataset"] in allow_corpora:
                relevant_annots = {}
                for role in allow_roles:
                    if role in instance["annotations"]:
                        relevant_annots[role] = instance["annotations"][role]
                    else:
                        relevant_annots[role] = len(instance["tokens"]) * ["O"]
                instance["annotations"] = relevant_annots
                self.data.append(instance)

            else:
                pass

        return None

    def SplitData(self) -> None:
        """Split the data loaded from file into subsets and store
        these subsets in the Data object.
        """

        self.split_data.clear()

        # to preserve the original order of the data,
        # shuffle a copy of the data only.
        cpy_data = copy.deepcopy(self.data)
        random.seed(10)
        random.shuffle(cpy_data)
        not_split = copy.deepcopy(cpy_data)

        for splt in self.splits:
            splt_point = int(splt * len(cpy_data))
            self.split_data.append(not_split[:splt_point])
            not_split = not_split[splt_point:]

        return None

    def conv2brown(self):
        """Convert the format of each annotation to the format of the brown corpus:
        [
            (this, "O"),
            ("is", "O"),
            ("a", "B),
            ("sentence", "I"),
            (".", ".")
        ]
        """

        # The unaltered data is spreserved as only the annotations
        # contained in the subsets are converted.
        for splt in self.split_data:
            for instance in splt:
                tokens = instance["tokens"]
                orig = instance["annotations"]
                brown = {}
                for label in orig:
                    brown[label] = []
                    for tup in zip(tokens, orig[label]):
                        # Set tag for full stop (".") to "."
                        # (necessary for training and predicting).
                        if tup[0] == ".":
                            brown[label].append((tup[0], "."))
                        else:
                            brown[label].append((tup[0].lower(), tup[1]))
                instance["annotations"] = brown
