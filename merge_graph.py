import sys
from tqdm import tqdm

org_file = sys.argv[1]
pre_file = sys.argv[2]

o = open("merged_graph.txt", "w")
#o.write("id,images\n")

d = {}
f = open(pre_file, "r")
lines = [line[:-1] for line in f.readlines()]
total = 0
for l in tqdm(lines):
    parts = l.split(",")
    q = parts[0]
    if len(parts) == 1 or parts[1] == "":
        continue
    candidates = parts[1].strip().split(" ")
    if q not in d:
        d[q] = candidates
f.close()


f = open(org_file, "r")
lines = [line[:-1] for line in f.readlines()]

test_and_index = {}
for l in lines:
    parts = l.split(",")
    q = parts[0]
    test_and_index[q] = 1

for l in tqdm(lines[:117577]): #merge queries first
    parts = l.split(",")
    q = parts[0]
    if q not in d:
        # not found any, just write it
        o.write(l + "\n")
        o.flush()
    else:
        # if overlap, write the line + non-overlapping
        candidates = parts[1].strip().split(" ")
        for c in range(0, len(candidates), 2):
            score = candidates[c+1]
            c = candidates[c]
            if c not in d[q]:
                d[q].append(c)
                d[q].append(score)
        o.write(q + ",")
        for k in d[q]:
            o.write(k + " ")
        o.write("\n")
        o.flush()
    total += 1


# write train image label node
for q in d:
    if q not in test_and_index:
        o.write(q + ",")
        for k in d[q]:
            o.write(k + " ")
        o.write("\n")
        o.flush()
        total += 1

print("total number of non-index images: " + str(total))
# merge index
for l in tqdm(lines[117577:]):
    parts = l.split(",")
    q = parts[0]
    if q not in d:
        # not found any, just write it
        o.write(l + "\n")
        o.flush()
    else:
        # if overlap, write the line + non-overlapping
        candidates = parts[1].strip().split(" ")
        for c in range(0, len(candidates), 2):
            score = candidates[c+1]
            c = candidates[c]
            if c not in d[q]:
                d[q].append(c)
                d[q].append(score)
        o.write(q + ",")
        for k in d[q]:
            o.write(k + " ")
        o.write("\n")
        o.flush()
