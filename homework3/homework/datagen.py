import json
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def check_candidate(model: CoTModel, text: str, truth: float):
    """
    Look at one generated completion.
    Returns (clean_text, parsed_answer, reason). reason == "ok" means we can keep it.
    """
    # The answer must be enclosed in <answer></answer> tags.
    if "<answer>" not in text or "</answer>" not in text:
        return text.strip(), float("nan"), "missing_answer_tag"  # no tags, or cut off too early

    # Drop anything the model wrote after the closing tag.
    clean = text.split("</answer>")[0] + "</answer>"
    pred = model.parse_answer(clean)
    if pred != pred:  # NaN
        return clean.strip(), pred, "not_a_number"
    if not is_answer_valid(pred, truth):
        return clean.strip(), pred, "wrong_answer"
    return clean.strip(), pred, "ok"

# def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
#     """
#     For every training question:
#       1. sample `oversample` chain-of-thought completions from the CoT model
#       2. keep the correct one (closest to the truth) that is in the right format
#       3. skip the question if none is correct
#     Saves a json list of [question, true_answer, reasoning].
#     """
#     checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
#     batch_size = 2  # questions per generate call (each one makes `oversample` sequences)
#     save_every = 50  # save progress every this many batches
#     # debug = N: only look at the first N questions, print every candidate with the reason it
#     # was accepted or rejected, and do NOT write the file. Try 20. Use 0 for the full run.
#     debug = 0

#     model = CoTModel(checkpoint)
#     data = Dataset("train")

#     n = len(data) if debug <= 0 else min(debug, len(data))
#     rows = []
#     reasons = Counter()  # counts over all candidates

#     out = Path(output_json)
#     if debug <= 0:
#         out.parent.mkdir(parents=True, exist_ok=True)

#     def save():
#         with open(out, "w") as f:
#             json.dump(rows, f, indent=2)

#     batches = range(0, n, batch_size)
#     for step, start in enumerate(tqdm(batches, desc="Generating RFT data")):
#         # Step 1: a few questions at a time, turned into proper chat prompts
#         batch = [data[i] for i in range(start, min(start + batch_size, n))]
#         prompts = [model.format_prompt(question) for question, _ in batch]

#         # Step 2: several different completions per question (temperature > 0)
#         generations = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

#         # Step 3: keep the best correct candidate of each question
#         for (question, truth), candidates in zip(batch, generations):
#             best_text, best_err = None, float("inf")
#             for text in candidates:
#                 clean, pred, reason = check_candidate(model, text, truth)
#                 reasons[reason] += 1
#                 if debug > 0:
#                     print(f"[{reason}] Q: {question} | expected: {truth} | parsed: {pred}\n    -> {clean!r}")
#                 if reason == "ok" and abs(pred - truth) < best_err:
#                     best_text, best_err = clean, abs(pred - truth)
#             if best_text is not None:
#                 rows.append([question, truth, best_text])
#             elif debug > 0:
#                 print(f"*** REJECTED question (no correct candidate): {question} | expected: {truth}")

#         # Step 4: free cached GPU memory, and save progress now and then (the job is long)
#         if torch.backends.mps.is_available():
#             torch.mps.empty_cache()
#         elif torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         if debug <= 0 and (step + 1) % save_every == 0:
#             save()

#     if debug <= 0:
#         save()  # debug runs never write the file, so they cannot overwrite a real dataset

#     total = sum(reasons.values())
#     print("Why candidates were accepted / rejected:")
#     for reason, count in reasons.most_common():
#         print(f"  {reason:20s} {count:6d}  ({100 * count / total:.1f}%)")
#     where = "(debug run, nothing saved)" if debug > 0 else f"-> {out}"
#     print(f"Kept {len(rows)} / {n} questions ({100 * len(rows) / n:.1f}%) {where}")
#     if debug <= 0 and len(rows) < 950:
#         print("WARNING: fewer than ~950 examples. Set debug = 20 inside generate_dataset and read the rejected ones.")

def generate_dataset(output_json: str, oversample: int = 5, temperature: float = 0.6):

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model = CoTModel(checkpoint)
    data = Dataset("train")
    rows = []
    reasons = Counter()
    rejected = 0

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    failed = set(json.load(open("data/failed_indices.json")))
    progress = tqdm(data, desc="Generating RFT data")
    for i, (question, truth) in enumerate(progress):
        if i not in failed:
            continue
        prompt = model.format_prompt(question)
        candidates = model.batched_generate(
            [prompt], num_return_sequences=oversample, temperature=temperature, new_max_tokens=150
        )[0]

        best_text, best_err = None, float("inf")
        checked = []
        for text in candidates:
            clean, pred, reason = check_candidate(model, text, truth)
            checked.append((clean, pred, reason))
            reasons[reason] += 1
            if reason == "ok" and abs(pred - truth) < best_err:
                best_text, best_err = clean, abs(pred - truth)
        if best_text is not None:
            rows.append([question, truth, best_text])
            with open(out, "w") as f:  # save on the fly so interruptions lose nothing
                json.dump(rows, f, indent=2)
        else:
            rejected += 1
            counts = Counter(r for _, _, r in checked)
            clean, pred, reason = checked[0]
            progress.write(
                f"REJECTED | expected: {truth} | missing_answer_tag: {counts['missing_answer_tag']}, "
                f"not_a_number: {counts['not_a_number']}, wrong_answer: {counts['wrong_answer']}\n"
                f"  e.g. {reason} | parsed: {pred} | {clean!r}"
            )
        progress.set_postfix(accepted=len(rows), rejected=rejected)

    # with open(out, "w") as f:
    #     json.dump(rows, f, indent=2)

    total = sum(reasons.values())
    print("Why candidates were accepted / rejected:")
    for reason, count in reasons.most_common():
        print(f"  {reason:20s} {count:6d}  ({100 * count / total:.1f}%)")
    print(f"Kept {len(rows)} / {len(data)} questions -> {out}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)