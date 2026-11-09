import json
from collections import Counter


def find_failed(
    train_json: str = "data/train.json",
    rft_json: str = "data/rft.json",
    output_json: str = "data/failed_indices.json",
):
    """Save indices of train questions that have no accepted entry in rft_json."""
    train = json.load(open(train_json))
    accepted = Counter(row[0] for row in json.load(open(rft_json)))

    failed = []
    for i, (question, _) in enumerate(train):
        # Count-based match so duplicate questions are handled correctly.
        if accepted[question] > 0:
            accepted[question] -= 1
        else:
            failed.append(i)

    with open(output_json, "w") as f:
        json.dump(failed, f)
    print(f"{len(failed)} / {len(train)} questions failed -> {output_json}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train_json", default="data/train.json")
    p.add_argument("--rft_json", default="data/rft_run1.json")
    p.add_argument("--output_json", default="data/failed_indices.json")
    find_failed(**vars(p.parse_args()))
