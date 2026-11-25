import json

import modal

app = modal.App("SpaGate-YOLO")
vol = modal.Volume.from_name("SpaGate-YOLO")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # needed by opencv
    .pip_install("torch", "torchvision", "ultralytics", "pandas", "numpy", "psutil")
    # your latest local code, copied over the old files inside project.tar
    .add_local_dir("model", "/overlay/model", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_file("config.py", "/overlay/config.py")
    .add_local_file("train.py", "/overlay/train.py")
    .add_local_file("fog_training.py", "/overlay/fog_training.py")
    .add_local_file("evaluate.py", "/overlay/evaluate.py")
    .add_local_file("logger.py", "/overlay/logger.py")
    .add_local_file("datasets/dataset_class.py", "/overlay/datasets/dataset_class.py")
)

ONE_STAGE = [["tone", "kernel", "defog", "homomorphic", "histeq"]]
ONE_STAGE_RESIDUAL = [["tone", "kernel", "defog", "homomorphic", "histeq", "residual"]]
ONE_STAGE_AOD = [["tone", "kernel", "defog", "homomorphic", "histeq", "aod"]]
TWO_STAGE = [["tone", "defog", "homomorphic", "histeq"], ["kernel", "tone"]]

#the four runs of the ablation, chosen with --variant
VARIANTS = {
    "A": {"stages": ONE_STAGE, "features": "0"},   #baseline: one stage, gates see no feature map
    "B": {"stages": ONE_STAGE, "features": "1"},   #spatial features only
    "C": {"stages": TWO_STAGE, "features": "0"},   #two stages only
    "D": {"stages": TWO_STAGE, "features": "1"},   #both
    "E": {"stages": ONE_STAGE_RESIDUAL, "features": "0"}, #A plus the learned residual network as a sixth filter
    "F": {"stages": ONE_STAGE_AOD, "features": "0"},      #A plus the AOD-style filter as a sixth filter, no residual network
}


def aug_label(aug):
    #short label for the run name: noaug, aug (fog, sensor and geom), aug+color+haze (those three and the optional groups), or for example aug-fog+geom
    if aug == "all":
        aug = "fog,sensor,geom"
    groups = sorted(g for g in aug.split(",") if g in ("fog", "sensor", "geom", "haze", "color", "wsensor"))
    if not groups:
        return "noaug"
    standard = {"fog", "sensor", "geom"}
    if standard <= set(groups):
        return "aug" + "".join("+" + g for g in groups if g not in standard)
    return "aug-" + "+".join(groups)


def unpack_dataset_archive(archive, target, work_dir):
    #unpacks a .zip or .tar of the combined dataset to target, with or without a top-level folder inside the archive:
    #the folder that holds annotations/ is taken as the dataset
    import os, shutil
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)
    shutil.unpack_archive(archive, work_dir) #zip or tar, from the extension
    root = work_dir
    for dirpath, dirnames, _ in os.walk(work_dir): #top down, so the shallowest match is found first
        if "annotations" in dirnames:
            root = dirpath
            break
    else:
        raise FileNotFoundError(f"no annotations folder found inside {archive}")
    shutil.move(root, target)
    shutil.rmtree(work_dir, ignore_errors=True)


@app.function(
    image=image,
    gpu=["A100-80GB", "H100"],   # tries the A100 first, falls back to H100
    cpu=8,                        # more cores so data loading keeps the GPU busy
    timeout=8 * 60 * 60,          # 8 hours, a ceiling only (you pay for the time actually used), the sync in `finally` does not run if modal kills the container
    volumes={"/data": vol},
)
def run(variant: str = "D", seeds: str = "42", aug: str = "fog,sensor,geom", yolo: str = "none", preproc: str = "1", epochs: int = 30, dataset: str = "fog_416", sampling: str = "none"):
    import os, sys, shutil, tarfile, subprocess, threading, time

    assert variant in VARIANTS, f"variant must be one of {list(VARIANTS)}"
    assert dataset in ("fog_416", "combined"), "dataset must be fog_416 or combined"
    assert sampling in ("none", "rare"), "sampling must be none or rare"
    v = VARIANTS[variant]

    proj = "/tmp/proj"
    if not os.path.exists(proj):
        print("extracting...", flush=True)
        with tarfile.open("/data/project.tar") as t:
            t.extractall(proj)

    if dataset == "combined" and not os.path.isdir(f"{proj}/datasets/combined_dataset"):
        # the combined dataset is not inside project.tar, so it comes from the Volume: the folder /datasets/combined_dataset,
        # or an archive of it (.zip or .tar) at /datasets/combined_dataset.zip, or at the root of the Volume
        target = f"{proj}/datasets/combined_dataset"
        volume_folder = "/data/datasets/combined_dataset"
        archives = [f"{where}/combined_dataset.{ext}" for where in ("/data/datasets", "/data") for ext in ("zip", "tar")]
        found = [a for a in archives if os.path.exists(a)]
        if os.path.isdir(volume_folder):
            print(f"copying the combined dataset from {volume_folder} ...", flush=True)
            shutil.copytree(volume_folder, target)
        elif found:
            print(f"extracting the combined dataset from {found[0]} ...", flush=True)
            unpack_dataset_archive(found[0], target, f"{proj}/_unpack")
        else:
            raise FileNotFoundError("datasets/combined_dataset is not in project.tar and the Volume has none of: " + ", ".join([volume_folder] + archives))

    # overwrite the old files from the tar with your current local versions
    shutil.copytree("/overlay", proj, dirs_exist_ok=True)
    print("overlaid:", sorted(os.listdir("/overlay")), sorted(os.listdir("/overlay/model")), flush=True)

    # show which batch size and lr the code will actually use
    subprocess.run(
        "grep -n 'DET_BATCH_SIZE\\|DET_NUM_EPOCHS\\|DET_LR \\|YOLO_WEIGHTS\\|STAGE_FILTERS' config.py",
        shell=True, cwd=proj,
    )

    # confirm the param_maker fix is in the code that will run
    subprocess.run("grep -n 'head.bias.copy_' model/param_maker.py", shell=True, cwd=proj)

    results_src = os.path.join(proj, "results")
    results_dst = "/data/results"

    def sync():
        # copy results to the Volume and persist them
        if os.path.exists(results_src):
            shutil.copytree(results_src, results_dst, dirs_exist_ok=True)
        vol.commit()

    # background thread: save progress to the Volume every 5 minutes
    stop = threading.Event()

    def periodic_sync():
        while not stop.wait(300):
            try:
                sync()
                print("[sync] results saved to volume", flush=True)
            except Exception as e:
                print(f"[sync] failed: {e}", flush=True)

    threading.Thread(target=periodic_sync, daemon=True).start()

    os.chdir(proj)
    env = {
        **os.environ,
        "PYTHONPATH": proj,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "STAGE_FILTERS": json.dumps(v["stages"]),   # read by config.py
        "GATE_USE_FEATURES": v["features"],
        "SEEDS": seeds,
        "YOLO_TRAIN": yolo,                          # none, head or all, read by config.py
        "EPOCHS": str(epochs),                       # read by config.py
        "PREPROC": preproc,                          # 0 trains the detector with no preprocessor
        "AUG": aug,                                  # augmentation groups, read by config.py, "none" turns them off
        "DATASET": dataset,                          # fog_416 or combined, read by config.py
        "SAMPLING": sampling,                        # none or rare, read by config.py
        "RUN_TAG": f"{variant}_{aug_label(aug)}",
    }
    print(f"variant {variant}: stages {v['stages']} | gate features {v['features']} | seeds {seeds} | aug {aug} | yolo {yolo} | preproc {preproc} | epochs {epochs} | dataset {dataset} | sampling {sampling}", flush=True)

    start = time.time()
    try:
        subprocess.run([sys.executable, "-u", "fog_training.py"], check=True, env=env)
    finally:
        stop.set()
        sync()  # final save, even if training crashed
        print(f"total time: {(time.time() - start) / 3600:.2f} h", flush=True)


@app.local_entrypoint()
def main(variant: str = "D", seeds: str = "42", aug: str = "fog,sensor,geom", yolo: str = "none", preproc: str = "1", epochs: int = 30, dataset: str = "fog_416", sampling: str = "none"):
    run.remote(variant, seeds, aug, yolo, preproc, epochs, dataset, sampling)