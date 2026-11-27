import json
from pathlib import Path


def validate_json():
    data_dir = Path(__file__).parent
    captions_balanced = json.load(open(data_dir / "data" / "valid_grader" / "all_mc_qas.json"))
    captions_generated = []
    captions_files = list(data_dir.glob("data/valid/*_captions.json"))

    for caption_file in captions_files:
        with open(caption_file) as f:
            cap = json.load(f)
            captions_generated.extend(cap)

    print(f"Number of captions golden: {len(captions_balanced)}")
    print(f"Number of captions generated: {len(captions_generated)}")

    generated_by_image = {}
    for cg in captions_generated:
        image_file = cg["image_file"]
        if image_file not in generated_by_image:
            generated_by_image[image_file] = []
        generated_by_image[image_file].append(cg["caption"])
    
    count_missing = 0
    count_correct = 0

    for idx, cb in enumerate(captions_balanced):
        image_file = cb["image_file"]
        correct_caption = cb["candidates"][cb["correct_index"]]

        if image_file not in generated_by_image:
            print(f"Not found: {cb}")
            count_missing += 1
            continue

        if correct_caption in generated_by_image[image_file]:
            count_correct += 1
        else:
            print(f"Wrong answer fpr image {image_file}:\nExpected: {correct_caption}\nGenerated: {generated_by_image[image_file]}")
        
    print(f"Number of missing images: {count_missing}")
    print(f"Number of correct caption matches: {count_correct} of {len(captions_balanced)}")


if __name__ == "__main__":
    validate_json()
