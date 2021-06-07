def counter(countables):
    counts = dict()
    for k in countables:
        if not k in list(counts.keys()):
            counts[k] = 1
        else:
            counts[k] += 1
    return counts


def conv2span(brown):
    spans = []
    span = {}
    tags = [tup[1] for tup in brown]
    for i in range(len(tags)):
        tag = tags[i]  # IOB tag
        if i == 0:
            span[i] = tag
        elif tag == "I" and span[i - 1] in ["B", "I"]:
            span[i] = tag
        elif tag == "O" and span[i - 1] == "O":
            span[i] = tag
        else:
            spans.append(span)
            span = {}
            span[i] = tag

        if i == len(tags) - 1:
            spans.append(span)
        else:
            pass
    return spans
