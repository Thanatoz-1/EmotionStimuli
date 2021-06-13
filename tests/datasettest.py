import json

with open("data/rectified-unified-with-offsets.json", "r") as file:
    data = json.load(file)
"""
reman = list()

for instance in data:
    if instance["dataset"] == "reman":
        reman.append(instance)
errors = list()
for instance in reman:
    if "noemo" in instance["emotions"] and len(instance["annotations"]) != 0:
        print(instance["annotations"].keys())"""

annots = {}
labels = set()
counter = {}

c = set()
for instance in data:
    if instance["dataset"] == "reman":
        for label in instance["annotations"]:
            c.add(label)
        # l += len(instance["tokens"])
print("#####################", c)

for instance in data:
    # if instance["dataset"] == "reman":
    if instance["dataset"] in counter:
        pass
    else:
        counter[instance["dataset"]] = {}
    for label in instance["annotations"]:
        if instance["dataset"] == "reman" and "cue" in label:
            lbl = "cue"
        else:
            lbl = label

        if lbl in counter[instance["dataset"]]:
            id = instance["id"]
            counter[instance["dataset"]][lbl][0] += 1
            counter[instance["dataset"]][lbl][1] += len(instance["tokens"])
        else:
            counter[instance["dataset"]][lbl] = [1, len(instance["tokens"])]

for item in counter:
    print("\n", item)
    for label in counter[item]:
        print(label)
        print(counter[item][label][0])
        print(counter[item][label][1] / counter[item][label][0])

with open("datatests.txt", "w") as write_file:
    for instance in data:
        if instance["dataset"] == "emotion-stimulus":

            # if "experiencer" in instance["annotations"]:
            #    if set(instance["annotations"]["experiencer"]) != {"O"}:
            write_file.write(instance["id"] + " " + instance["text"] + "\n")

        """
        annots[instance["id"]] = {}
        for label in instance["annotations"]:
            tmp = set(instance["annotations"][label])
            if tmp != {"O"}:
                annots[instance["id"]][label] = []

                for i in range(len(instance["annotations"][label])):
                    if instance["annotations"][label][i] != "O":
                        annots[instance["id"]][label].append(i)
            else:
                annots.pop(instance["id"])
                break

for inst in annots:
    if len(annots[inst]) == 4:

        for label in annots[inst]:

            test = [annots[inst][label] for label in annots[inst]]
            # print("test", test)

            counter = 0
            for a in test:
                for b in test:
                    if a == b:
                        counter += 1
                    else:
                        pass
            if counter == len(test):
                match = True
            else:
                match = False

        if match:
            for instance in data:
                if (
                    instance["id"] == inst
                    and len(instance["tokens"]) > 15
                    and len(instance["tokens"]) < 20
                ):
                    pass
                    # print(inst)"""
