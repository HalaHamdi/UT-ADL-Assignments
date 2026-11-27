from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info
import json


HORIZONTAL_PHRASE = {"left": "left of", "right": "right of"}
VERTICAL_PHRASE = {"front": "in front of", "back": "behind"}

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """

    captions = []
    karts =extract_kart_objects(info_path=info_path, view_index=view_index, img_width=img_width, img_height=img_height)
    track_name = extract_track_info(info_path)
    ego_kart = next((k for k in karts if k["is_center_kart"]), None)
    # 1. Ego car
    # {kart_name} is the ego car.
    if ego_kart:
        captions.append(f"{ego_kart['kart_name']} is the ego car.")
   

    # 2. Counting
    # There are {num_karts} karts in the scenario.
    captions.append(f"There are {len(karts)} karts in the scene.")


    # 3. Track name
    # The track is {track_name}.
    captions.append(f"The track is {track_name}.")


    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    if ego_kart:
        for kart in karts:
            ego_x, ego_y = ego_kart["center"]
            if kart is not ego_kart:
                kart_x, kart_y = kart["center"]
                name = kart["kart_name"]
                horizontal = "left" if kart_x <= ego_x else "right"
                vertical = "front" if kart_y < ego_y else "back"

                h = HORIZONTAL_PHRASE[horizontal]
                v = VERTICAL_PHRASE[vertical]

                captions.append(f"{name} is {h} the ego car.")
                captions.append(f"{name} is {v} the ego car.")                

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()



def generate(split: str = "train", data_dir: str = "data"):
    """
    Build the CLIP training file: loop over every *_info.json in data_dir/split, every view,
    and save all (image_file, caption) pairs to data_dir/split/generated_captions.json.

    Usage (from the homework4 folder):
        python -m homework.generate_captions generate
    """
    out = []
    for info_path in sorted(Path(data_dir, split).glob("*_info.json")):
        base = info_path.stem.replace("_info", "")
        with open(info_path) as f:
            n_views = len(json.load(f)["detections"])
        for v in range(n_views):
            image_name = f"{base}_{v:02d}_im.jpg"
            if not Path(data_dir, split, image_name).exists():
                continue
            for caption in generate_caption(str(info_path), v):
                out.append({"image_file": f"{split}/{image_name}", "caption": caption})

    out_path = Path(data_dir, split, "generated_captions.json")
    out_path.write_text(json.dumps(out))
    print(f"Wrote {len(out)} captions to {out_path}")
"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate})


if __name__ == "__main__":
    main()
