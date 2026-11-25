"""Letterbox every XWOD image to 416 x 416, adjust the YOLO labels, keep only your 5 classes,
and merge the training and validation sets into one training set.

Run from the project folder:
    python resize_xwod.py
    python resize_xwod.py --src datasets/XWOD_dataset.v1.1 --dst datasets/XWOD_resized --size 416

The new folder has the same layout as the old one (train.v1.1/images, train.v1.1/labels, test.v1.1/images, ...).
- Images are letterboxed like fog_416: the longer side becomes 416, the picture is centred, the rest is grey 128.
- Labels stay in YOLO format (class, x centre, y centre, width, height, between 0 and 1) but are recomputed
  for the padded image, and the class ids are changed to your order: person, car, bus, bicycle, motorcycle.
- The class names are read from the yaml in the source folder. Truck boxes are removed.
- The training and validation folders are combined into ONE folder that keeps the name of the training folder.
  Because both sets had their own numbering (fog_train_00001 and fog_valid_00001 and so on), every merged image is
  renamed <weather>_train_<number> and numbered again from 1 for each weather prefix, with no gaps:
  the original training images come first, then the original validation images.
  Images that are removed (only trucks) are removed before numbering, so the numbers stay continuous.
  The old name of every merged image is written to rename_map.csv in the new folder.
  The test folder (any other folder) keeps its original names.
- A new yaml with the 5 classes is written next to the data, the original is kept as <name>.original.yaml.
  There is no separate validation folder any more, so "val" in the yaml points to the merged folder too
  (only so that the yaml stays valid). Make your own train/val split from the merged folder.
Needs PyYAML (pip install pyyaml, it is already there if ultralytics is installed).
"""
import argparse
import csv
import hashlib
import json
import random
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

TARGET_NAMES = ["person", "car", "bus", "bicycle", "motorcycle"] #same order as CLASS_NAMES in config.py
try:
    from config import CLASS_NAMES #use the project's own list if it can be imported
    TARGET_NAMES = list(CLASS_NAMES)
except Exception:
    pass

ALIASES = { #XWOD class name (lower case) -> your class name
    "person": "person", "pedestrian": "person", "people": "person",
    "car": "car",
    "bus": "bus",
    "bicycle": "bicycle", "bike": "bicycle",
    "motorcycle": "motorcycle", "motorbike": "motorcycle",
}
DROP_NAMES = {"truck"} #removed
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8) #np.fromfile also works with unusual characters in the path on Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR) #bgr, None if the file is broken


def write_image(path, image):
    suffix = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in (".jpg", ".jpeg") else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    encoded.tofile(str(path))


def letterbox(image, size, pad_value):
    height, width = image.shape[:2]
    scale = size / max(height, width) #the longer side becomes exactly size
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR #area averaging is best when shrinking
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_x = (size - new_w) // 2 #left bar, the right bar gets the extra pixel if the difference is odd
    pad_y = (size - new_h) // 2 #top bar
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, new_w, new_h, pad_x, pad_y


def read_source_names(yaml_data):
    names = yaml_data.get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def parse_overrides(text):
    #"5:bicycle,2:drop" -> {5: 3, 2: -1}
    result = {}
    for item in [t for t in text.split(",") if t.strip()]:
        cid, value = item.split(":")
        value = value.strip().lower()
        if value in ("drop", "-1"):
            result[int(cid)] = -1
        elif value in TARGET_NAMES:
            result[int(cid)] = TARGET_NAMES.index(value)
        else:
            raise SystemExit(f"--map: '{value}' is not one of {TARGET_NAMES} or 'drop'")
    return result


def build_class_map(source_names, overrides):
    class_map, unknown = dict(overrides), []
    for cid, name in source_names.items():
        if cid in class_map:
            continue
        key = name.strip().lower()
        if key in DROP_NAMES:
            class_map[cid] = -1
        elif ALIASES.get(key) in TARGET_NAMES:
            class_map[cid] = TARGET_NAMES.index(ALIASES[key])
        else:
            unknown.append((cid, name))
    if unknown:
        raise SystemExit(f"these class names are not recognised: {unknown}. "
                         f"Give them with --map, for example --map \"{unknown[0][0]}:person\" (targets: {TARGET_NAMES} or drop)")
    return class_map


def convert_label_line(line, size, new_w, new_h, pad_x, pad_y, class_map):
    #returns (converted line or None, status), status is ok, dropped or unreadable
    parts = line.split()
    if len(parts) < 5:
        return None, "unreadable"
    try:
        cid = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None, "unreadable"
    target = class_map.get(cid)
    if target is None:
        return None, "unreadable" #a class id that is not in the yaml
    if target < 0:
        return None, "dropped"

    if len(values) == 4: #a box: x centre, y centre, width, height
        xc, yc, w, h = values
        xc = (xc * new_w + pad_x) / size
        yc = (yc * new_h + pad_y) / size
        w = w * new_w / size
        h = h * new_h / size
        x1, x2 = np.clip([xc - w / 2, xc + w / 2], 0.0, 1.0) #clip in case of tiny rounding overflow
        y1, y2 = np.clip([yc - h / 2, yc + h / 2], 0.0, 1.0)
        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        return f"{target} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}", "ok"

    if len(values) % 2 == 0: #a polygon: x1 y1 x2 y2 ...
        out = []
        for i in range(0, len(values), 2):
            x = np.clip((values[i] * new_w + pad_x) / size, 0.0, 1.0)
            y = np.clip((values[i + 1] * new_h + pad_y) / size, 0.0, 1.0)
            out.append(f"{x:.6f} {y:.6f}")
        return f"{target} " + " ".join(out), "ok"
    return None, "unreadable"


def weather_of(stem):
    match = re.match(r"^(?P<weather>.+?)_(?:train|test|valid|val)_\d+", stem) #dust_test_00001 -> dust
    return match.group("weather") if match else "unknown"


def split_role(folder_name):
    #train.v1.1 -> train, valid.v1.1 / val -> valid, everything else (test.v1.1) -> other
    lower = folder_name.lower()
    if lower.startswith("train"):
        return "train"
    if lower.startswith("val"):
        return "valid"
    return "other"


def keeps_boxes(label_path, class_map):
    #True if the image still has at least one box after the class mapping (or never had any boxes).
    #Same rule that is used when the image is written, but it needs no image, so the numbering can be planned first.
    if not label_path.exists():
        return True
    had_boxes = False
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        had_boxes = True
        _, status = convert_label_line(line, 1, 1, 1, 0, 0, class_map)
        if status == "ok":
            return True
    return not had_boxes


def build_plan(splits, class_map, keep_truck_only, counts):
    #Decides, before any image is written, which images are kept and what every output file is called.
    #Returns (plan, merged_name). plan: output folder name -> list of (image path, source label folder, new stem, source folder name)
    train_like = [s for s in splits if split_role(s.name) in ("train", "valid")]
    train_like.sort(key=lambda s: (split_role(s.name) != "train", s.name)) #training folders first, then validation
    others = [s for s in splits if split_role(s.name) == "other"]
    merged_name = train_like[0].name if train_like else None #the merged folder keeps the training folder's name

    plan = {}
    numbering = Counter() #weather prefix -> last number used, shared by the training and validation images
    for split in train_like + others:
        merged = split in train_like
        out_name = merged_name if merged else split.name
        label_dir = split / "labels"
        images = sorted(p for p in (split / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        label_stems = {p.stem for p in label_dir.glob("*.txt")} if label_dir.is_dir() else set()
        counts[out_name]["labels without image"] += len(label_stems - {p.stem for p in images})

        for image_path in images:
            label_path = label_dir / (image_path.stem + ".txt")
            if not keep_truck_only and not keeps_boxes(label_path, class_map):
                counts[out_name]["images dropped (only trucks)"] += 1 #nothing left to detect, and it gets no number
                continue
            if merged:
                weather = weather_of(image_path.stem)
                numbering[weather] += 1
                new_stem = f"{weather}_train_{numbering[weather]:05d}"
            else:
                new_stem = image_path.stem #other folders keep their names
            plan.setdefault(out_name, []).append((image_path, label_dir, new_stem, split.name))
    return plan, merged_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="datasets/XWOD_dataset.v1.1")
    parser.add_argument("--dst", default="datasets/XWOD_resized")
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--pad", type=int, default=128, help="grey value of the bars, fog_416 uses 128")
    parser.add_argument("--map", default="", help="class overrides, for example \"5:bicycle,2:drop\"")
    parser.add_argument("--keep-truck-only", action="store_true",
                        help="keep images that had only truck boxes (they become images with no boxes), the default drops them")
    parser.add_argument("--preview", type=int, default=12, help="how many images to save with their boxes drawn, 0 for none")
    parser.add_argument("--force", action="store_true", help="redo everything")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"source folder not found: {src}")

    yaml_files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in (".yaml", ".yml"))
    if not yaml_files:
        raise SystemExit("no yaml file in the source folder, so the class names are unknown")
    yaml_path = yaml_files[0]
    yaml_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    source_names = read_source_names(yaml_data)
    class_map = build_class_map(source_names, parse_overrides(args.map))

    print(f"class names from {yaml_path.name}:")
    for cid, name in sorted(source_names.items()):
        target = class_map[cid]
        print(f"  {cid} {name:12s} -> " + ("removed" if target < 0 else f"{target} {TARGET_NAMES[target]}"))
    mapped_targets = {t for t in class_map.values() if t >= 0}
    for t, name in enumerate(TARGET_NAMES):
        if t not in mapped_targets:
            print(f"  WARNING: no XWOD class maps to {name}")

    splits = sorted(p for p in src.iterdir() if p.is_dir() and (p / "images").is_dir())
    if not splits:
        raise SystemExit(f"no folders with an images subfolder found in {src}")

    counts = defaultdict(Counter) #output folder -> counters
    plan, merged_name = build_plan(splits, class_map, args.keep_truck_only, counts)

    if merged_name:
        parts = Counter(job[3] for job in plan.get(merged_name, []))
        merged_from = [s.name for s in splits if split_role(s.name) in ("train", "valid")]
        print(f"\nmerging {' + '.join(f'{n} ({parts[n]} images)' for n in merged_from)} into {merged_name} "
              f"({sum(parts.values())} images), renamed <weather>_train_<number> and numbered again")
    else:
        print("\nWARNING: no training or validation folder found, nothing to merge")

    plan_text = "\n".join(f"{out}|{src_name}|{p.name}|{stem}" for out, jobs in plan.items() for p, _, stem, src_name in jobs)
    settings = {"version": 3, "size": args.size, "pad": args.pad, "class_map": {str(k): v for k, v in class_map.items()},
                "targets": TARGET_NAMES, "keep_truck_only": args.keep_truck_only,
                "merge_train_valid": True, "plan": hashlib.md5(plan_text.encode("utf-8")).hexdigest()}
    marker = dst / ".resize_settings.json"
    force = args.force
    if dst.exists() and not force:
        old = json.loads(marker.read_text()) if marker.exists() else None
        if old != settings:
            print("\nthe settings or the source files differ from the earlier run (or there is no record of it), so everything is redone")
            force = True
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir(): #other files go across unchanged
        if item.is_file() and item not in yaml_files:
            shutil.copy2(item, dst / item.name)
    for item in yaml_files: #the yaml with the 5 classes
        shutil.copy2(item, dst / f"{item.stem}.original{item.suffix}")
    new_yaml = dict(yaml_data)
    new_yaml["names"] = list(TARGET_NAMES)
    new_yaml["nc"] = len(TARGET_NAMES)
    if merged_name and yaml_data.get("train") is not None:
        for key in ("val", "valid"): #there is no separate validation folder any more
            if key in new_yaml:
                new_yaml[key] = yaml_data["train"]
    (dst / yaml_path.name).write_text(yaml.safe_dump(new_yaml, sort_keys=False), encoding="utf-8")
    for item in yaml_files[1:]:
        shutil.copy2(item, dst / item.name)
    print(f"\nwrote {yaml_path.name} with {len(TARGET_NAMES)} classes: {TARGET_NAMES}")
    if merged_name:
        print(f"note: val in the yaml now points to the merged folder, make your own train/val split from it")

    if force: #remove old outputs, so no image from an earlier run is left over (also an old separate valid folder)
        for split in splits:
            if (dst / split.name).exists():
                shutil.rmtree(dst / split.name)
        if (dst / "_preview").exists():
            shutil.rmtree(dst / "_preview")

    if merged_name: #old name -> new name for every renamed image
        with open(dst / "rename_map.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["new_folder", "new_image", "source_folder", "source_image"])
            for image_path, _, new_stem, source_split in plan[merged_name]:
                writer.writerow([merged_name, new_stem + image_path.suffix, source_split, image_path.name])

    per_weather = defaultdict(Counter) #output folder -> weather -> images kept
    boxes_per_class = defaultdict(Counter) #output folder -> class name -> boxes
    preview_pool = []

    for out_name, jobs in plan.items():
        out_images = dst / out_name / "images"
        out_labels = dst / out_name / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for n, (image_path, label_dir, new_stem, source_split) in enumerate(jobs, 1):
            out_image = out_images / (new_stem + image_path.suffix)
            out_label = out_labels / (new_stem + ".txt")
            if out_image.exists() and out_label.exists() and not force:
                counts[out_name]["skipped (already done)"] += 1
                per_weather[out_name][weather_of(new_stem)] += 1
                preview_pool.append((out_image, out_label))
                continue

            image = read_image(image_path)
            if image is None:
                counts[out_name]["unreadable images"] += 1 #its number stays unused
                print(f"WARNING: cannot read {image_path}, its number {new_stem} is skipped")
                continue
            canvas, new_w, new_h, pad_x, pad_y = letterbox(image, args.size, args.pad)

            lines_out = []
            label_path = label_dir / (image_path.stem + ".txt")
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    converted, status = convert_label_line(line, args.size, new_w, new_h, pad_x, pad_y, class_map)
                    if status == "ok":
                        lines_out.append(converted)
                        boxes_per_class[out_name][TARGET_NAMES[int(converted.split()[0])]] += 1
                    elif status == "dropped":
                        counts[out_name]["truck boxes removed"] += 1
                    else:
                        counts[out_name]["unreadable label lines"] += 1
            else:
                counts[out_name]["images without label file"] += 1

            write_image(out_image, canvas)
            out_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
            counts[out_name]["images kept"] += 1
            counts[out_name]["boxes kept"] += len(lines_out)
            per_weather[out_name][weather_of(new_stem)] += 1
            preview_pool.append((out_image, out_label))
            if n % 500 == 0:
                print(f"{out_name}: {n}/{len(jobs)}")

    marker.write_text(json.dumps(settings, indent=1))

    print("\n===== summary =====")
    for split_name, counter in counts.items():
        print(f"{split_name}: " + ", ".join(f"{k} {v}" for k, v in counter.items()))
        print("   boxes per class:", dict(boxes_per_class[split_name]))
        print("   images per weather prefix:", dict(sorted(per_weather[split_name].items())))

    if args.preview > 0 and preview_pool:
        random.seed(0)
        preview_dir = dst / "_preview"
        preview_dir.mkdir(exist_ok=True)
        for k, (img_path, lbl_path) in enumerate(random.sample(preview_pool, min(args.preview, len(preview_pool)))):
            image = read_image(img_path)
            size = image.shape[0]
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    p = line.split()
                    if len(p) == 5:
                        xc, yc, w, h = (float(v) for v in p[1:])
                        x1, y1 = int((xc - w / 2) * size), int((yc - h / 2) * size)
                        cv2.rectangle(image, (x1, y1), (int((xc + w / 2) * size), int((yc + h / 2) * size)), (0, 0, 255), 1)
                        cv2.putText(image, TARGET_NAMES[int(p[0])], (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            write_image(preview_dir / f"{k:02d}_{img_path.stem}.jpg", image)
        print(f"\nsaved {min(args.preview, len(preview_pool))} preview images with boxes and class names to {preview_dir}")
    print(f"\ndone, resized dataset is in {dst}")


if __name__ == "__main__":
    main()