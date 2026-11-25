import numpy as np
import pandas as pd
import torch

from config import (
    DEVICE, CLASS_NAMES, DET_NUM_CLASSES, DET_BATCH_SIZE, FOG_TEST_SPLITS, MAIN_SCORES,
    EVAL_CONF, EVAL_NMS_IOU, EVAL_MAX_DET, EVAL_IOU,
)
from datasets.dataset_class import make_test_loader


def box_iou_numpy(box, boxes):
    #box shape: (4,), boxes shape: (M, 4), all [x1, y1, x2, y2]
    inter_x1 = np.maximum(box[0], boxes[:, 0]) #corners of the overlap
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0) #zero when the boxes do not overlap
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / np.maximum(union_area, 1e-9) #shape: (M,)


def class_average_precision(detections, ground_truths, iou_threshold):
    #detections: list of (image_index, score, x1, y1, x2, y2) for one class, over all images
    #ground_truths: dict image_index -> array (K, 4) of the boxes of the same class, only images that have some
    num_ground_truths = 0 #total boxes that should be found
    for boxes in ground_truths.values():
        num_ground_truths += len(boxes)
    if num_ground_truths == 0:
        return None #this class does not appear in the split

    used = {} #image_index -> which ground truth boxes are already matched
    for image_index, boxes in ground_truths.items():
        used[image_index] = np.zeros(len(boxes), dtype=bool)

    detections = sorted(detections, key=lambda detection: detection[1], reverse=True) #most confident first

    true_positive = np.zeros(len(detections)) #1 where the detection found a new ground truth box
    false_positive = np.zeros(len(detections)) #1 where it did not
    for d, detection in enumerate(detections):
        image_index = detection[0]
        box = np.array(detection[2:6])

        matched = False
        if image_index in ground_truths:
            ious = box_iou_numpy(box, ground_truths[image_index]) #overlap with every ground truth box in this image
            best = int(np.argmax(ious)) #the closest ground truth box
            if ious[best] >= iou_threshold and not used[image_index][best]:
                used[image_index][best] = True #each ground truth box can only be found once
                matched = True

        if matched:
            true_positive[d] = 1
        else:
            false_positive[d] = 1

    cum_tp = np.cumsum(true_positive) #running totals down the confidence ranking
    cum_fp = np.cumsum(false_positive)
    recall = cum_tp / num_ground_truths
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    recall = np.concatenate([[0.0], recall, [1.0]]) #sentinels at both ends
    precision = np.concatenate([[0.0], precision, [0.0]])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1]) #precision never increases as recall grows

    changed = np.where(recall[1:] != recall[:-1])[0] #places where recall goes up
    ap = np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]) #area under the curve
    return float(ap)


def mean_average_precision(all_detections, all_ground_truths, iou_threshold=EVAL_IOU):
    #all_detections: one (n, 6) array per image, [x1, y1, x2, y2, confidence, class]
    #all_ground_truths: one (m, 5) array per image, [x1, y1, x2, y2, class]
    ap_per_class = {} #class name -> ap

    for class_id in range(DET_NUM_CLASSES):
        class_detections = [] #every detection of this class, over all images
        class_ground_truths = {} #image_index -> boxes of this class

        for image_index in range(len(all_ground_truths)):
            ground_truth = all_ground_truths[image_index]
            class_boxes = ground_truth[ground_truth[:, 4] == class_id][:, :4]
            if len(class_boxes) > 0:
                class_ground_truths[image_index] = class_boxes

            detection = all_detections[image_index]
            class_rows = detection[detection[:, 5] == class_id]
            for row in class_rows:
                class_detections.append((image_index, float(row[4]), row[0], row[1], row[2], row[3]))

        ap = class_average_precision(class_detections, class_ground_truths, iou_threshold)
        if ap is not None:
            ap_per_class[CLASS_NAMES[class_id]] = ap

    map_value = float(np.mean(list(ap_per_class.values()))) #mean over the classes that appear
    return map_value, ap_per_class


@torch.no_grad()
def collect_predictions(model, loader):
    #runs the model once over a split and keeps the raw detections, so the same predictions can be scored per split and pooled
    model.eval()

    all_detections = [] #one array per image
    all_ground_truths = [] #one array per image

    for images, boxes_list in loader:
        images = images.to(DEVICE)
        detections = model.detect(images, EVAL_CONF, EVAL_NMS_IOU, EVAL_MAX_DET) #one (n, 6) tensor per image

        for detection, boxes in zip(detections, boxes_list):
            all_detections.append(detection.float().cpu().numpy())
            all_ground_truths.append(boxes.numpy())

    return all_detections, all_ground_truths


def metrics_from_predictions(all_detections, all_ground_truths):
    map_value, ap_per_class = mean_average_precision(all_detections, all_ground_truths)

    metrics = {"map50": map_value}
    for class_name, ap in ap_per_class.items():
        metrics[f"ap50_{class_name}"] = ap
    return metrics


def evaluation(model, loader):
    all_detections, all_ground_truths = collect_predictions(model, loader)
    return metrics_from_predictions(all_detections, all_ground_truths)


def evaluate_splits(model, splits=FOG_TEST_SPLITS, batch_size=DET_BATCH_SIZE):
    #works for any of the splits (test_foggy, test_clean, rtts, and test_<weather> with the combined dataset),
    #so a trained model can also be tested on one split by itself
    model = model.to(DEVICE)

    test_rows = [] #one row per split, then one row per main score
    predictions = {} #split -> (detections, ground truths), kept to pool them for the main scores
    for split in splits:
        loader = make_test_loader(split, batch_size)
        all_detections, all_ground_truths = collect_predictions(model, loader)
        predictions[split] = (all_detections, all_ground_truths)
        metrics = metrics_from_predictions(all_detections, all_ground_truths)
        print(f"{split} | mAP@0.5 {metrics['map50']:.4f}")

        row = {"split": split}
        row.update(metrics)
        test_rows.append(row)

    #main scores: clean, foggy and bad weather, one mAP over all the images of the splits in MAIN_SCORES[name] together
    #a score is only given when every one of its splits was evaluated, so a partial list of splits never gives a misleading number
    main_values = {}
    for score_name, members in MAIN_SCORES.items():
        if len(members) == 0 or any(member not in predictions for member in members):
            continue
        pooled_detections, pooled_ground_truths = [], []
        for member in members:
            pooled_detections += predictions[member][0]
            pooled_ground_truths += predictions[member][1]
        metrics = metrics_from_predictions(pooled_detections, pooled_ground_truths)

        row = {"split": f"score_{score_name}"}
        row.update(metrics)
        test_rows.append(row)
        main_values[score_name] = metrics["map50"]

    if main_values:
        print("main scores mAP@0.5 | " + " | ".join(f"{name} {value:.4f}" for name, value in main_values.items()))

    return pd.DataFrame(test_rows) #one row per split and one per main score (score_clean, score_foggy, score_bad_weather), columns: split, map50, ap50 for each class