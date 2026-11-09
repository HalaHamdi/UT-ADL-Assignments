import json
from collections import defaultdict, deque
from pathlib import Path

DATA = Path(__file__).parent / "data"

train = json.load(open(DATA / "train.json"))
rft = json.load(open(DATA / "rft.json"))

# Queue per question so duplicate questions keep their relative order
pending = defaultdict(deque)
for entry in rft:
    pending[entry[0]].append(entry)

ordered = []
for question, *_ in train:
    if pending[question]:
        ordered.append(pending[question].popleft())

# Anything not found in train goes at the end
leftover = [e for q in pending.values() for e in q]
ordered.extend(leftover)

with open(DATA / "rft.json", "w") as f:
    json.dump(ordered, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"train={len(train)} rft_in={len(rft)} rft_out={len(ordered)} unmatched={len(leftover)}")
