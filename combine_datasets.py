"""Add the resized XWOD images to fog_416, so the result has the same structure as fog_416.

Run from the project folder (after resize_xwod.py):
    python combine_datasets.py
    python combine_datasets.py --fog datasets/fog_416 --xwod datasets/XWOD_resized --dst datasets/combined_dataset

What it does
- Copies fog_416 to the --dst folder (skipped when --dst is the same folder as --fog, then XWOD is added in place).
- XWOD images go to <dst>/train/<weather>/ and <dst>/test/<weather>/ next to clean/ and foggy/,
  for example train/dust/dust_train_00001.jpg. The weather is the prefix of the file name.
- XWOD has one YOLO label file per image (class, x centre, y centre, width, height, between 0 and 1).
  fog_416 has one annotation file per folder with one line per image:
      train/dust/dust_train_00001.jpg x1,y1,x2,y2,class x1,y1,x2,y2,class ...
  (pixels of the 416 x 416 image). So every weather gets its own file in <dst>/annotations/:
  train_dust.txt, test_dust.txt, train_rain.txt, ... The existing fog_416 annotation files are not touched.
- Class ids are matched by name: the XWOD ids (names in the XWOD yaml) are converted to the ids of fog_416
  (names in classes.names), so it also works if the two orders differ.
- image_meta.csv and split_manifest.json of fog_416 are NOT changed, the XWOD images are not listed in them.
"""
import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML is needed to read the class names: pip install pyyaml")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESERVED = {"clean", "foggy"} #folder names that fog_416 already uses inside train and test


def weather_of(stem):
    match = re.match(r"^(?P<weather>.+?)_(?:train|test|valid|val)_\d+", stem) #dust_train_00001 -> dust
    return match.group("weather") if match else "unknown"


def group_of(folder_name):
    #train.v1.1 and valid.v1.1 -> train (the valid images are normally already merged into the train folder), test.v1.1 -> test
    lower = folder_name.lower()
    if lower.startswith(("train", "val")):
        return "train"
    if lower.startswith("test"):
        return "test"
    return None


def copy_if_new(src, dst):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size == Path(src).stat().st_size:
        return dst #already copied, so a second run is quick
    return shutil.copy2(src, dst)


def read_fog_names(path):
    if not path.is_file():
        raise SystemExit(f"{path} not found, the class order of fog_416 is needed")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_xwod_names(xwod):
    yamls = sorted(p for p in xwod.iterdir() if p.suffix.lower() in (".yaml", ".yml") and ".original" not in p.stem)
    if not yamls:
        raise SystemExit(f"no yaml file in {xwod}, run resize_xwod.py first")
    names = yaml.safe_load(yamls[0].read_text(encoding="utf-8")).get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def to_pixel_box(values, size):
    #values: the numbers after the class id. Returns (x1, y1, x2, y2) in pixels or None if unreadable.
    if len(values) == 4: #x centre, y centre, width, height
        xc, yc, w, h = values
        x1, x2, y1, y2 = xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2
    elif len(values) >= 6 and len(values) % 2 == 0: #a polygon, use its bounding box
        x1, x2, y1, y2 = min(values[0::2]), max(values[0::2]), min(values[1::2]), max(values[1::2])
    else:
        return None
    def px(v):
        return int(min(max(np.floor(v * size + 0.5), 0), size - 1)) #rounded, inside 0 .. size-1 like fog_416
    return px(x1), px(y1), px(x2), px(y2)


def convert_label(label_path, size, id_map):
    #returns (boxes as "x1,y1,x2,y2,class" strings, number of boxes in the file, number of bad boxes)
    boxes, total, bad = [], 0, 0
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if not parts:
                continue
            total += 1
            try:
                cid = int(float(parts[0]))
                values = [float(v) for v in parts[1:]]
            except ValueError:
                bad += 1
                continue
            box = to_pixel_box(values, size)
            if box is None or cid not in id_map or box[2] <= box[0] or box[3] <= box[1]:
                bad += 1 #unreadable, unknown class, or no area left after rounding
                continue
            boxes.append(f"{box[0]},{box[1]},{box[2]},{box[3]},{id_map[cid]}")
    return boxes, total, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fog", default="datasets/fog_416")
    parser.add_argument("--xwod", default="datasets/XWOD_resized")
    parser.add_argument("--dst", default="datasets/combined_dataset", help="same as --fog to add XWOD to fog_416 itself")
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--skip-empty", action="store_true", help="leave out XWOD images that have no boxes")
    args = parser.parse_args()

    fog, xwod, dst = Path(args.fog), Path(args.xwod), Path(args.dst)
    for folder in (fog, xwod):
        if not folder.is_dir():
            raise SystemExit(f"folder not found: {folder}")

    fog_names = read_fog_names(fog / "classes.names")
    xwod_names = read_xwod_names(xwod)
    lookup = {n.lower(): i for i, n in enumerate(fog_names)}
    id_map = {}
    print("class ids (XWOD -> fog_416):")
    for xid, name in sorted(xwod_names.items()):
        if name.lower() not in lookup:
            raise SystemExit(f"XWOD class '{name}' is not in {fog / 'classes.names'}: {fog_names}")
        id_map[xid] = lookup[name.lower()]
        print(f"  {xid} {name} -> {id_map[xid]}")

    in_place = dst.resolve() == fog.resolve()
    if not in_place:
        print(f"\ncopying {fog} to {dst} ...")
        shutil.copytree(fog, dst, dirs_exist_ok=True, copy_function=copy_if_new)
    (dst / "annotations").mkdir(parents=True, exist_ok=True)

    splits = sorted(p for p in xwod.iterdir() if p.is_dir() and (p / "images").is_dir() and group_of(p.name))
    for p in xwod.iterdir():
        if p.is_dir() and (p / "images").is_dir() and not group_of(p.name):
            print(f"WARNING: folder {p.name} is not train, valid or test, so it is skipped")
    if not splits:
        raise SystemExit(f"no train or test folders with images found in {xwod}")

    lines = defaultdict(list) #(group, weather) -> annotation lines
    counts = defaultdict(Counter) #(group, weather) -> counters
    boxes_per_class = Counter()
    checked = 0
    for split in splits:
        group = group_of(split.name)
        images = sorted(p for p in (split / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for n, image_path in enumerate(images, 1):
            weather = weather_of(image_path.stem)
            if weather in RESERVED:
                raise SystemExit(f"weather name '{weather}' clashes with the existing fog_416 folder of the same name")
            key = (group, weather)
            boxes, total, bad = convert_label(split / "labels" / (image_path.stem + ".txt"), args.size, id_map)
            counts[key]["bad boxes skipped"] += bad
            if total > 0 and not boxes:
                counts[key]["images skipped (no usable box)"] += 1
                continue
            if total == 0 and args.skip_empty:
                counts[key]["images skipped (no boxes)"] += 1
                continue

            if checked < 20: #the conversion to pixels assumes the images are already --size x --size
                data = np.fromfile(str(image_path), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is not None and image.shape[:2] != (args.size, args.size):
                    raise SystemExit(f"{image_path} is {image.shape[1]}x{image.shape[0]}, not {args.size}x{args.size}, run resize_xwod.py first")
                checked += 1

            out_dir = dst / group / weather
            out_dir.mkdir(parents=True, exist_ok=True)
            copy_if_new(image_path, out_dir / image_path.name)
            lines[key].append(" ".join([f"{group}/{weather}/{image_path.name}"] + boxes))
            counts[key]["images"] += 1
            counts[key]["boxes"] += len(boxes)
            for box in boxes:
                boxes_per_class[fog_names[int(box.rsplit(",", 1)[1])]] += 1
            if n % 2000 == 0:
                print(f"{split.name}: {n}/{len(images)}")

    for (group, weather), entries in sorted(lines.items()):
        (dst / "annotations" / f"{group}_{weather}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")

    print("\n===== summary =====")
    for (group, weather), counter in sorted(counts.items()):
        print(f"{group}_{weather}.txt: " + ", ".join(f"{k} {v}" for k, v in counter.items()))
    print("boxes per class:", dict(boxes_per_class))
    print(f"\ndone, combined dataset is in {dst}")


if __name__ == "__main__":
    main()