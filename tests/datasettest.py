import json

with open("data/rectified-unified-with-offsets.json", "r") as file:
    data = json.load(file)

reman = list()

for instance in data:
    if instance["dataset"] == "reman":
        reman.append(instance)
errors = list()
for instance in reman:
    if "noemo" in instance["emotions"] and len(instance["annotations"]) != 0:
        print(instance["annotations"].keys())
