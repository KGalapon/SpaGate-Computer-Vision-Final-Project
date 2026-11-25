import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent 
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"


# which dataset to use, env DATASET: fog_416 (default) or combined (fog_416 plus the XWOD bad weather images, datasets/combined_dataset)
DATASET = os.environ.get("DATASET", "fog_416")
assert DATASET in ("fog_416", "combined"), "DATASET must be fog_416 or combined"
DATASET_FOLDERS = {"fog_416": "fog_416", "combined": "combined_dataset"}

# fog dataset
FOG_DIR = DATASETS_DIR / DATASET_FOLDERS[DATASET]
FOG_SPLIT_FILES = {
    "train": "train_clean.txt",
    "test_foggy": "test_foggy.txt",
    "test_clean": "test_clean.txt",
    "rtts": "rtts_test.txt",
}
BASE_TEST_SPLITS = ["test_foggy", "test_clean", "rtts"] #the fog_416 test splits


def find_weather_names(prefix):
    #the combined dataset has one annotation file per weather event (train_dust.txt, test_dust.txt, ...), they are found by name
    annotations_dir = FOG_DIR / "annotations"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"{annotations_dir} not found, DATASET={DATASET} needs the folder {FOG_DIR}")
    names = []
    for path in sorted(annotations_dir.glob(f"{prefix}_*.txt")):
        name = path.stem[len(prefix) + 1:] #train_dust -> dust
        if name not in ("clean", "foggy"): #those two are the fog_416 files
            names.append(name)
    return names


WEATHER_TRAIN_SPLITS = [] #train_<weather>, added to the training set, only with the combined dataset
WEATHER_TEST_SPLITS = []  #test_<weather>, one test split per weather event, only with the combined dataset
if DATASET == "combined":
    for _weather in find_weather_names("train"):
        FOG_SPLIT_FILES[f"train_{_weather}"] = f"train_{_weather}.txt"
        WEATHER_TRAIN_SPLITS.append(f"train_{_weather}")
    for _weather in find_weather_names("test"):
        FOG_SPLIT_FILES[f"test_{_weather}"] = f"test_{_weather}.txt"
        WEATHER_TEST_SPLITS.append(f"test_{_weather}")
    assert WEATHER_TRAIN_SPLITS and WEATHER_TEST_SPLITS, f"no train_<weather>.txt or test_<weather>.txt files in {FOG_DIR / 'annotations'}"
FOG_TEST_SPLITS = BASE_TEST_SPLITS + WEATHER_TEST_SPLITS #every split that is evaluated

# the three main scores, each one is the mAP over all the images of the listed test splits together
# add "rtts" to bad_weather to count it as bad weather too, it is reported on its own either way
MAIN_SCORES = {
    "clean": ["test_clean"],
    "foggy": ["test_foggy"],
    "bad_weather": WEATHER_TEST_SPLITS, #all the weather events of the combined dataset, empty (so not reported) with fog_416
}
FOG_BETAS = ["0.05", "0.06", "0.07", "0.08", "0.09", "0.10", "0.11", "0.12", "0.13", "0.14"]
FOGGY_PROB = 2 / 3
NUM_WORKERS = 6

# augmentation, train split only. groups: fog, sensor, geom. env AUG="fog,sensor" picks groups, AUG=none turns all off
_aug_text = os.environ.get("AUG", "fog,sensor,geom")
if _aug_text == "all":
    _aug_text = "fog,sensor,geom" #"all" means all three groups
AUG_GROUPS = {g for g in _aug_text.split(",") if g in ("fog", "sensor", "geom")}
AUG_FOG = "fog" in AUG_GROUPS       #our own depth-dependent fog rendered from the clean image
AUG_SENSOR = "sensor" in AUG_GROUPS #blur, noise and jpeg on the foggy images
AUG_GEOM = "geom" in AUG_GROUPS     #random crop and random affine on every training image

# depth-dependent fog: t = exp(-tau), tau grows from the near field (bottom) to the far field (top), I = J * t + A * (1 - t)
AUG_CUSTOM_FOG_PROB = 0.5           #share of the foggy samples that get our own fog, the rest keep the pre-rendered fog
AUG_FOG_TAU_NEAR = (0.0, 0.5)       #optical thickness at the bottom of the image, 0 is no fog at all
AUG_FOG_TAU_FAR = (1.0, 3.0)        #optical thickness at the top, 3.0 is t = 0.05, a very dense far field
AUG_FOG_GRADIENT_GAMMA = (0.7, 1.5) #shape of the vertical gradient, below 1 the fog spreads further down
AUG_FOG_NOISE_AMOUNT = (0.0, 0.4)   #how much of the depth comes from a smooth random map instead of the vertical gradient
AUG_AIRLIGHT_RANGE = (0.5, 0.85)    #brightness of the fog colour
AUG_AIRLIGHT_TINT = 0.05            #random tint added to each colour channel of the fog colour

# sensor effects, applied to foggy samples after the fog
AUG_BLUR_PROB = 0.3
AUG_BLUR_SIGMA = (0.3, 1.2)
AUG_NOISE_PROB = 0.3
AUG_NOISE_SIGMA = (0.0, 0.02)       #gaussian noise, in image units (0 to 1)
AUG_JPEG_PROB = 0.3
AUG_JPEG_QUALITY = (40, 90)

# geometric, applied to every training image
AUG_CROP_PROB = 0.5
AUG_CROP_SIDE = (0.6, 1.0)          #fraction of the image side that is kept, then resized back to full size
AUG_AFFINE_PROB = 0.5
AUG_AFFINE_SCALE = (0.6, 1.15)      #below 1 zooms out, objects get smaller, the new border is filled with the letterbox grey
AUG_AFFINE_ROTATE = 5.0             #degrees, plus or minus
AUG_AFFINE_SHEAR = 3.0              #degrees, plus or minus
AUG_AFFINE_TRANSLATE = 0.1          #fraction of the image size, plus or minus
AUG_BOX_MIN_SIDE = 2.0              #boxes thinner than this many pixels after a transform are dropped
AUG_BOX_MIN_VISIBLE = 0.3           #boxes with less than this fraction still inside the image are dropped

# detector
CLASS_NAMES = ["person", "car", "bus", "bicycle", "motorcycle"]
DET_NUM_CLASSES = 5
YOLO_WEIGHTS = "yolov8n.pt" #smaller detector, was yolov3u.pt
DET_BATCH_SIZE = 32
DET_NUM_EPOCHS = int(os.environ.get("EPOCHS", "30")) #env EPOCHS overrides it
DET_LR = 3e-4 #was 1e-3, the weak detector gives much larger gradients
DET_LR_MIN = 1e-6
DET_WEIGHT_DECAY = 0
DET_GRAD_CLIP = 1.0 #was 10.0, tighter so one large-gradient step cannot swing the gates

# which parts of the detector are trained, YOLO_TRAIN: none = frozen (what every run so far did), head = only the Detect layer, all = the whole detector
YOLO_TRAIN = os.environ.get("YOLO_TRAIN", "none")
assert YOLO_TRAIN in ("none", "head", "all"), "YOLO_TRAIN must be none, head or all"
YOLO_LR_MULT = 0.3                                    #the detector trains at this fraction of DET_LR, a guess, pretrained weights want a gentler lr
PREPROC_ON = os.environ.get("PREPROC", "1") == "1"    #PREPROC=0 trains the detector alone, with no preprocessor, to see what fine-tuning gives by itself

# evaluation
EVAL_CONF = 0.001
EVAL_NMS_IOU = 0.6
EVAL_MAX_DET = 100
EVAL_IOU = 0.5

# contrast stretch
CONTRAST_NUM_SEGMENTS = 11

# local kernel
KERNEL_SIZE = 9
KERNEL_SCALE = 0.1

# defog
DEFOG_PATCH_SIZE = 15
DEFOG_MIN_TRANSMISSION = 0.1
DEFOG_MIN_AIRLIGHT = 0.1
DEFOG_TOP_FRACTION = 0.001
DEFOG_MAX_OMEGA = 0.95
DEFOG_INIT_RAW = -4.0

# homomorphic
HOMO_PAD = 32
HOMO_CUTOFF_RANGE = (0.01, 0.3)
HOMO_GAIN_LOW_RANGE = (0.3, 1.5)
HOMO_GAIN_HIGH_RANGE = (0.7, 2.5)
HOMO_INIT_CUTOFF = 0.05
HOMO_INIT_GAIN_LOW = 1.0
HOMO_INIT_GAIN_HIGH = 1.0

# residual network filter (learned, per pixel)
RESIDUAL_CHANNELS = 32   #width of its hidden layers
RESIDUAL_SCALE = 0.5     #largest change it can make to a pixel value, per colour channel

# AOD-style filter: J = K(x) * x - K(x) + b, K is predicted per pixel, K = 1 / t for a fog model with white airlight
AOD_CHANNELS = 8       #width of its hidden layers
AOD_K_MAX = 4.0        #K can go from 1 / AOD_K_MAX to AOD_K_MAX, 4 covers a transmission down to 0.25
AOD_B_RANGE = 0.3      #b can go from 1 - AOD_B_RANGE to 1 + AOD_B_RANGE, it plays the role of the fog colour

# histogram equalisation
HIST_NUM_BINS = 256

# encoder
EMB = 64
ENCODER_INPUT_SIZE = 256
ENCODER_CHANNELS = [3, 16, 32, 32, 32, 32]
LEAKY_SLOPE = 0.1

# stages: filters inside a stage run in parallel and are gated, stages run one after another
# names: tone, kernel, defog, homomorphic, histeq
# old single-stage model: [["tone", "kernel", "defog", "homomorphic", "histeq"]] with GATE_USE_FEATURES = False
STAGE_FILTERS = json.loads(os.environ["STAGE_FILTERS"]) if "STAGE_FILTERS" in os.environ else [["tone", "defog", "homomorphic", "histeq"], ["kernel", "tone"]] #env STAGE_FILTERS (json) overrides it

# parameter maker
PARAM_HIDDEN_DIM = 128

# gate
GATE_CHANNELS = 16
GATE_SCALE = 0.5
GATE_DETACH_FILTER = True #changed from False: the gate no longer sends gradient into the filters, cleaner credit assignment
GATE_USE_FEATURES = os.environ.get("GATE_USE_FEATURES", "1") == "1" #the gates also get the encoder's spatial feature map, env GATE_USE_FEATURES=0 turns it off
GATE_OUT_BIAS = 0.05 #no longer used by gate.py, which now starts every gate at 0.2 (GATE_INIT_PROB in gate.py), kept because spagate.py still imports it

# training
DEVICE = 'cuda'
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42").split(",")] #env SEEDS=42,43,44 overrides it
RUN_TAG = os.environ.get("RUN_TAG", "local") #label of the run, goes into the results folder name