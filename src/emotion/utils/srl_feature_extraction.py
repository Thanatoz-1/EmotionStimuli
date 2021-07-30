# from .practnlptools import Annotator


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
