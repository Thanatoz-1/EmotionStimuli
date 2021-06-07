__author__ = "Maximilian Wegge"
from .file_reading import Data


class Dataset:
    """The Dataset object stores the subset of a Data object
    as Instance objects.
    """

    def __init__(self, data: Data, splt: int = 0) -> None:
        """Initialize attributes of Dataset object and import instances from data.

        Args:
            data (Data): Data object containing the input data.
            splt (int, optional): The index of the subset of the Data object.
            Defaults to 0.
        """

        self.instances = {}  # Instace objects stored by corresponding id
        self.roles = set()  # metadata
        self.corpora = set()  # metadata
        self.LoadData(data, splt)

    def LoadData(self, data: Data, splt: int = 0) -> None:
        """Convert the data from a Data object to Instance objects.

        Args:
            data (Data): Data object containing the input data.
            splt (int, optional): The index of the subset of the Data object.
            Defaults to 0.
        """
        source = data.split_data[splt]
        for i in range(len(source)):
            src_inst = source[i]
            src_corpus = src_inst["dataset"]
            src_tokens = src_inst["tokens"]
            src_annotations = src_inst["annotations"]
            src_id = src_inst["id"]

            # create Instance object with relevant data
            instance = Instance(tokens=src_tokens, corpus=src_corpus)
            for emo_role in src_annotations:
                instance.set_gld(role=emo_role, annotation=src_annotations[emo_role])
                # set metadata for Dataset object
                self.roles.add(emo_role)
                self.corpora.add(src_corpus)
                # store Instance object in Dataset object
                self.instances[src_id] = instance

        return None


class Instance:
    """The Instance object stores the relevant data
    for each instance of the data (tokens, gold and predicted
    annotations, metadata).
    """

    def __init__(self, tokens: list, corpus: str):
        """Initialize Instance object.

        Args:
            tokens (list): tokens in the instance.
            corpus (str): corpus where this instances appears in.
        """
        self.tokens = tokens
        self.gold = {}  # gold annotations, stored by emotion role
        self.pred = {}  # gold annotations, stored by emotion role
        self.roles = set()  # metadata
        self.corpus = corpus  # metadata

    def set_gld(self, role: str, annotation: list) -> None:
        """Assign gold annotation for emotion role to this instance.

        Args:
            role (str): Emotion role of the annotation.
            annotation (list): Annotation (IOB-tag sequence).
        """
        self.roles.add(role)
        self.gold[role] = annotation
        return None

    def get_tokens(self) -> list:
        """Return a list of all tokens of this instance"""
        return self.tokens

    def get_roles(self) -> set:
        """Return set containing all emotion roles this instance is annotated with."""
        return self.roles

    def get_gld_annots(self) -> dict:
        """Return the gold annotations of this instance."""
        return self.gold_annots

    def get_prd_annots(self) -> dict:
        """Return the predicted annotations of this instance."""
        return self.pred_annots
