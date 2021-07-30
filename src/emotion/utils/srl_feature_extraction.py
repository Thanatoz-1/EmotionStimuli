__author__ = "Maximilian Wegge"

# from .practnlptools import Annotator
# This is a standalone file which we ran once to generate the SRL features.
# As the import depends on other frameworl, it has been deprecated.
# Please download and use the external library to use this module.


def getsrl(srl_input):
    """Function to extract the SRL features from dataset

    Args:
        srl_input (dict): contains the dictionaey of SRL features with token_ids
    """
    annot = Annotator()
    srl_output = {}
    errors = set([])
    for idx in srl_input:
        sent = srl_input[idx]
        if sent[-1] != ".":
            sent += "."
        try:
            srl_analysis = annot.getAnnotations(sent)["srl"]
            srl_output[idx] = srl_analysis
        except:
            errors.add(idx)

    return errors, srl_output
