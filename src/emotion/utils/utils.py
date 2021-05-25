def counter(countables):
    counts = dict()
    for k in countables:
        if not k in list(counts.keys()):
            counts[k] = 1
        else:
            counts[k] += 1
    return counts
