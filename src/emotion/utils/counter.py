__author__ = "Tushar Dhyani"


def counter(countables):
    """
    Counter for counting the values inside a particular list or dict.
    This is just a scratch/vanilla version of collections.Counter

    Args:
    countables: List of countables to be counted.
    """
    counts = dict()
    for k in countables:
        if not k in list(counts.keys()):
            counts[k] = 1
        else:
            counts[k] += 1
    return counts
