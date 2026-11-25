import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.optim as optim

from config import (
    RESULTS_DIR, DEVICE,
    DET_BATCH_SIZE, DET_NUM_EPOCHS, DET_LR, DET_LR_MIN, DET_WEIGHT_DECAY, DET_GRAD_CLIP,
)
from datasets.dataset_class import make_train_loader
from logger import logger
from evaluate import evaluate_splits


MB = 1024 ** 2
NUM_DIAG_BATCHES = 8 #fixed batches used for the diagnostics, averaged so one noisy batch cannot mislead
GATE_LR_MULT = 0.1 #gates train at this fraction of the lr, they saturate fast with Adam
GATE_NAMES = ["tone", "kernel", "defog", "homo", "histeq"]


def make_yolo_batch(boxes_list, images):
    #boxes_list: one (N, 5) tensor per image, [x1, y1, x2, y2, class] in pixels
    #the ultralytics loss wants every box in one long list, with normalised [x_center, y_center, w, h]
    height, width = images.shape[2], images.shape[3] #image size, used to normalise the boxes

    batch_idx_list = [] #which image in the batch each box belongs to
    cls_list = [] #class of each box
    bboxes_list = [] #normalised [x_center, y_center, w, h] of each box

    for image_index, boxes in enumerate(boxes_list):
        num_boxes = boxes.shape[0] #boxes in this image
        if num_boxes > 0:
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            class_ids = boxes[:, 4]

            x_center = (x1 + x2) / 2 / width #centre of the box, 0 to 1
            y_center = (y1 + y2) / 2 / height
            box_width = (x2 - x1) / width #size of the box, 0 to 1
            box_height = (y2 - y1) / height

            batch_idx_list.append(torch.full((num_boxes,), float(image_index)))
            cls_list.append(class_ids)
            bboxes_list.append(torch.stack([x_center, y_center, box_width, box_height], dim=1))

    if len(bboxes_list) > 0:
        batch_idx = torch.cat(batch_idx_list, dim=0) #shape: (M,)
        cls = torch.cat(cls_list, dim=0) #shape: (M,)
        bboxes = torch.cat(bboxes_list, dim=0) #shape: (M, 4)
    else:
        batch_idx = torch.zeros(0) #a batch where no image has a box
        cls = torch.zeros(0)
        bboxes = torch.zeros(0, 4)

    batch = {
        "batch_idx": batch_idx.to(images.device),
        "cls": cls.to(images.device),
        "bboxes": bboxes.to(images.device),
    }
    return batch


def train_model(model, seed, model_name, lr=DET_LR, batch_size=DET_BATCH_SIZE, num_epochs=DET_NUM_EPOCHS):

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(DEVICE)

    train_loader = make_train_loader(seed, batch_size)

    print(f"steps/epoch: {len(train_loader)} | batch_size: {batch_size} | lr: {lr}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "No trainable parameters: check requires_grad"
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")

    gate_ids = {id(p) for p in model.preproc.gate_parameters()} #the gate parameters, they get a lower lr
    gate_params = [p for p in trainable if id(p) in gate_ids]
    other_params = [p for p in trainable if id(p) not in gate_ids]
    optimizer = optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": gate_params, "lr": lr * GATE_LR_MULT},
    ], weight_decay=DET_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=DET_LR_MIN) #lr goes down smoothly over the epochs

    folder = f"run_{model_name}_{seed}"
    log = logger(folder, RESULTS_DIR) #writes train_log.csv in results/<folder>
    log.save()

    MODEL_FOLDER = Path(folder)
    CHECKPOINT_DIR = Path(RESULTS_DIR) / MODEL_FOLDER / Path('checkpoints/')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    process = psutil.Process(os.getpid())

    diag_data = [] #fixed batches kept on the cpu, so the diagnostics are comparable between epochs
    for _, (d_imgs, d_boxes) in zip(range(NUM_DIAG_BATCHES), train_loader):
        diag_data.append((d_imgs, d_boxes))

    def diag_eval(use_preproc):
        #averages over the fixed batches: detector loss, |enhanced - input|, gate mean, gate std, defog omega
        total_loss, total_diff = 0.0, 0.0
        gate_mean_sum, gate_std_sum, omega_sum = 0.0, 0.0, 0.0
        for d_imgs, d_boxes in diag_data:
            d_imgs = d_imgs.to(DEVICE)
            if use_preproc:
                enh, G = model.enhance(d_imgs)
                omega_sum += model.preproc.defog_omega(d_imgs)
                gate_mean_sum = gate_mean_sum + G.mean(dim=(0, 2, 3)).cpu().numpy()
                gate_std_sum = gate_std_sum + G.std(dim=(2, 3)).mean(dim=0).cpu().numpy() #spread across pixels, 0 means spatially flat
                total_diff += (enh - d_imgs).abs().mean().item()
            else:
                enh = d_imgs
            d_batch = make_yolo_batch(d_boxes, enh)
            d_batch["img"] = enh
            d_loss, _ = model.yolo.loss(d_batch)
            total_loss += d_loss.sum().item() / enh.shape[0]
        n = len(diag_data)
        return total_loss / n, total_diff / n, gate_mean_sum / n, gate_std_sum / n, omega_sum / n

    model.eval()
    with torch.no_grad():
        baseline_loss = diag_eval(False)[0] #loss with no preproc at all
    model.train()
    print("gate order:", model.preproc.gate_names, flush=True)
    print(f"identity baseline loss on {NUM_DIAG_BATCHES} fixed batches: {baseline_loss:.4f}", flush=True)

    for epoch in range(num_epochs):

        model.train()
        train_loss_sum, n_train = 0, 0
        gate_grad_sum = torch.zeros(len(model.preproc.gate_modules()), device=DEVICE) #sum of grad norms of each gate's output conv

        epoch_lr = optimizer.param_groups[0]["lr"] #the lr used during this epoch

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        for batch_idx, (images, boxes_list) in enumerate(train_loader):

            images = images.to(DEVICE)
            batch = make_yolo_batch(boxes_list, images) #boxes in the format the yolo loss wants

            n_train += images.size(0)

            train_loss = model.compute_loss(images, batch) #filters, then yolo, then the loss

            optimizer.zero_grad()
            train_loss.backward()
            for gi, gate in enumerate(model.preproc.gate_modules()):
                if gate.out.weight.grad is not None:
                    gate_grad_sum[gi] += gate.out.weight.grad.norm() #exactly 0 means the gate is dead
            torch.nn.utils.clip_grad_norm_(trainable, DET_GRAD_CLIP) #stops one bad batch from blowing up the weights
            optimizer.step()

            train_loss_sum += train_loss.item() * images.size(0)

        scheduler.step()

        current_epoch = epoch + 1
        train_loss = train_loss_sum / n_train

        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start

        peak = torch.cuda.max_memory_allocated() / MB if use_cuda else 0.0
        ram_mb = process.memory_info().rss / MB

        update_dict = {
            'epoch': [current_epoch],
            'train_loss': [train_loss],
            's/epoch': [epoch_time],
            'lr': [epoch_lr],
            'batch_size': [batch_size],
            'vram_use': [peak],
            'ram_use': [ram_mb]
        }

        log.update(update_dict)

        print(
            f"Epoch {current_epoch}/{num_epochs} | "
            f"train {train_loss:.4f} | "
            f"lr {epoch_lr:.2e} | "
            f"{epoch_time:.1f}s | VRAM {peak:.0f}MB | RAM {ram_mb:.0f}MB"
        )

        #diagnostic: are the gates alive, and does the preproc change the image at all, on the fixed batches
        model.eval()
        with torch.no_grad():
            enh_loss, enh_diff, gate_mean, gate_std, omega = diag_eval(True)
            print("gate mean:", gate_mean.round(4))
            print("gate std :", gate_std.round(4))
            print("gate grad:", (gate_grad_sum / len(train_loader)).cpu().numpy().round(6))
            print("mean |enhanced - input|:", round(enh_diff, 5), "| defog omega:", round(omega, 4))
            print(f"fixed-batch loss: enhanced {enh_loss:.4f} | identity {baseline_loss:.4f}", flush=True)
        model.train()

        torch.save(model.preproc.state_dict(), CHECKPOINT_DIR / "preproc_last.pt") #yolo is frozen and can be reloaded from yolov3u.pt, so only the preproc is saved, no validation set so we always keep the last epoch

    test_metrics = evaluate_splits(model, batch_size=batch_size) #mAP@0.5 on test_foggy, test_clean and rtts, reported separately
    log.save_test(test_metrics)
    return test_metrics