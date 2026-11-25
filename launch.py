import argparse

import modal

#python launch.py A                                    variant A, seed 42, all augmentations, frozen detector
#python launch.py A 42 none                            the old pipeline without augmentation
#python launch.py E 42 all --epochs 60                60 epochs instead of 30
#python launch.py A 42 none --yolo all                 preprocessor and the whole detector trained together
#python launch.py A 42 none --yolo all --preproc 0     the detector alone, no preprocessor (the control for the line above)
#python launch.py A --dataset combined                 train on fog_416 + the XWOD bad weather images (default is --dataset fog_416)
parser = argparse.ArgumentParser()
parser.add_argument("variant")
parser.add_argument("seeds", nargs="?", default="42")
parser.add_argument("aug", nargs="?", default="fog,sensor,geom")
parser.add_argument("--yolo", default="none", choices=["none", "head", "all"])
parser.add_argument("--preproc", default="1", choices=["0", "1"])
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", default="fog_416", choices=["fog_416", "combined"]) #combined = datasets/combined_dataset
args = parser.parse_args()

f = modal.Function.from_name("SpaGate-YOLO", "run")
call = f.spawn(args.variant, args.seeds, args.aug, args.yolo, args.preproc, args.epochs, args.dataset)
print("started", call.object_id, vars(args))