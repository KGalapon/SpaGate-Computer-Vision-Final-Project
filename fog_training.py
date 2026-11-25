#!/usr/bin/env python
# coding: utf-8

# fog_416: SpaGate + frozen YOLO
# Dataset: `datasets/dataset_class.py`. Train function and evaluation: `train.py`. Model: `model/`.

import json

import pandas as pd
import torch
import torch.nn as nn

from config import (
    SEEDS, FOG_TEST_SPLITS, DEVICE, RESULTS_DIR, RUN_TAG, DATASET, MAIN_SCORES,
    YOLO_WEIGHTS, STAGE_FILTERS, GATE_USE_FEATURES, GATE_DETACH_FILTER, AUG_GROUPS,
    YOLO_TRAIN, YOLO_LR_MULT, PREPROC_ON,
    DET_LR, DET_BATCH_SIZE, DET_NUM_EPOCHS, DET_GRAD_CLIP,
)
from datasets.dataset_class import FogDataset, make_train_loader, make_test_loaders
from model.spagate import SpaGate
from model.yolo import FogYolo
from train import train_model, GATE_LR_MULT
from evaluate import evaluate_splits


class IdentityPreproc(nn.Module): #does nothing, used to measure the detector on its own, and to train the detector with no preprocessor
    gate_names = []

    def forward(self, x):
        return x, torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

    def gate_modules(self):
        return []

    def gate_parameters(self):
        return []

    def defog_omega(self, x):
        return float("nan")


def build_model(seed):
    torch.manual_seed(seed)
    preproc = SpaGate() if PREPROC_ON else IdentityPreproc() #PREPROC=0 trains the detector alone
    return FogYolo(preproc)


#one name for the whole run, it ends up in the results folder: results/run_<run_name>_<seed>
detector = YOLO_WEIGHTS.replace(".pt", "")
extra = (f"_yolo-{YOLO_TRAIN}" if YOLO_TRAIN != "none" else "") + ("" if PREPROC_ON else "_nopreproc") + (f"_ep{DET_NUM_EPOCHS}" if DET_NUM_EPOCHS != 30 else "") + ("" if DATASET == "fog_416" else f"_{DATASET}") #what differs from the standard runs
run_name = f"spagate_{detector}_{RUN_TAG}{extra}_lr{DET_LR:g}"
print(f"run name: {run_name} | seeds: {SEEDS}")
print(f"dataset: {DATASET} | test splits: {FOG_TEST_SPLITS}")
print(f"main scores: " + " | ".join(f"{name} = {splits if splits else 'not used with this dataset'}" for name, splits in MAIN_SCORES.items()))
print(f"stages: {STAGE_FILTERS} | gate features: {GATE_USE_FEATURES} | augmentation groups: {sorted(AUG_GROUPS)}")
print(f"preproc on: {PREPROC_ON} | detector training: {YOLO_TRAIN} (lr x {YOLO_LR_MULT})")

out_dir = RESULTS_DIR / f"summary_{run_name}"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "config.json", "w") as f: #so the folder says which run it was, even if it is renamed later
    json.dump({
        "run_name": run_name, "dataset": DATASET, "test_splits": FOG_TEST_SPLITS, "main_scores": MAIN_SCORES, "detector": YOLO_WEIGHTS, "stage_filters": STAGE_FILTERS,
        "gate_use_features": GATE_USE_FEATURES, "gate_detach_filter": GATE_DETACH_FILTER,
        "lr": DET_LR, "gate_lr_mult": GATE_LR_MULT, "batch_size": DET_BATCH_SIZE,
        "epochs": DET_NUM_EPOCHS, "grad_clip": DET_GRAD_CLIP, "seeds": SEEDS, "aug_groups": sorted(AUG_GROUPS), "preproc_on": PREPROC_ON, "yolo_train": YOLO_TRAIN, "yolo_lr_mult": YOLO_LR_MULT,
    }, f, indent=2)


# Look at the data

train_set = FogDataset("train", use_flip=True) #fog_416 images: 2/3 foggy, 1/3 clean every time an image is loaded, the weather images of the combined dataset are used as they are
image, boxes = train_set[0]
num_weather_train = len(train_set.can_fog) - sum(train_set.can_fog) #real bad weather images, 0 with fog_416
print(f"train images: {len(train_set)} (fog_416 {len(train_set) - num_weather_train}, bad weather {num_weather_train}) | image: {tuple(image.shape)} | boxes: {tuple(boxes.shape)}")

for split in FOG_TEST_SPLITS:
    print(f"{split}: {len(FogDataset(split))} images")

#train_model builds its own loaders, this is only a check that they work
train_loader = make_train_loader(SEEDS[0])
test_loaders = make_test_loaders()
images, boxes_list = next(iter(train_loader))
print(f"images: {tuple(images.shape)} | boxes in the first image: {tuple(boxes_list[0].shape)}")
print(f"steps/epoch: {len(train_loader)}")
for split, loader in test_loaders.items():
    print(f"{split}: {len(loader)} steps")


# Build the model, and the baselines (once, they do not depend on the seed)

model = build_model(SEEDS[0]).to(DEVICE)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable params: {trainable_params:,} | frozen params: {frozen_params:,}")

#the real baseline: the frozen yolo with no preproc at all, this is the number training has to beat
trained_preproc = model.preproc
model.preproc = IdentityPreproc()
identity_metrics = evaluate_splits(model)
model.preproc = trained_preproc
print("identity baseline (no preproc):")
print(identity_metrics.to_string(index=False), flush=True)

#the untrained preproc is not the identity: every gate starts at 0.2 and some filters (histeq) change the image
if PREPROC_ON:
    untrained_metrics = evaluate_splits(model)
    print("untrained preproc:")
    print(untrained_metrics.to_string(index=False), flush=True)
    untrained_metrics.to_csv(out_dir / "untrained_preproc.csv", index=False)

identity_metrics.to_csv(out_dir / "identity_baseline.csv", index=False)


# Train and test, one model per seed

all_metrics = []
for i, seed in enumerate(SEEDS):
    if i > 0:
        model = build_model(seed) #fresh preproc and detector for every seed
    test_metrics = train_model(model, seed, run_name) #trains, then reports mAP@0.5 on test_foggy, test_clean and rtts
    test_metrics = test_metrics.copy()
    test_metrics.insert(0, "seed", seed)
    all_metrics.append(test_metrics)
    print(f"seed {seed} done")
    print(test_metrics.to_string(index=False), flush=True)

    pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "per_seed_test_metrics.csv", index=False) #saved after every seed


# Summary over seeds

results = pd.concat(all_metrics, ignore_index=True)
metric_cols = [c for c in results.columns if c.startswith("map50") or c.startswith("ap50")]
summary = results.groupby("split")[metric_cols].agg(["mean", "std"])
summary.to_csv(out_dir / "summary_over_seeds.csv")
print(f"mean and std over {len(SEEDS)} seed(s):")
print(summary["map50"].round(4).to_string())

#to test one split by itself after loading saved preproc weights (yolo is frozen, so it is not in the checkpoint):
#model.preproc.load_state_dict(torch.load(f"results/run_{run_name}_42/checkpoints/preproc_last.pt", map_location="cuda"))
#print(evaluate_splits(model, splits=["rtts"]).to_string(index=False))