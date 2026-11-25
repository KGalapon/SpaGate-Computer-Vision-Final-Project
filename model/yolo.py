import copy

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

try:
    from ultralytics.utils.nms import non_max_suppression #newer ultralytics versions
except ImportError:
    from ultralytics.utils.ops import non_max_suppression #older ultralytics versions

from config import DET_NUM_CLASSES, CLASS_NAMES, YOLO_WEIGHTS


class FogYolo(nn.Module): #the filter pipeline in front of a frozen YOLOv3 detector, only the filter pipeline is trained

    def __init__(self, preproc, num_classes=DET_NUM_CLASSES):
        super().__init__()
        self.preproc = preproc #GatedPreproc, image -> enhanced image

        pretrained = YOLO(YOLO_WEIGHTS).model #coco pretrained yolov3u, 80 classes
        yaml_dict = copy.deepcopy(pretrained.yaml) #the architecture description
        self.yolo = DetectionModel(yaml_dict, nc=num_classes, verbose=False) #same architecture with our number of classes
        self.yolo.load(pretrained, verbose=False) #copy every weight whose shape still fits, the class layers are still new

        coco_ids = [] #the coco class number of each of our classes, in our order
        for class_name in CLASS_NAMES:
            for coco_id, coco_name in pretrained.names.items():
                if coco_name == class_name:
                    coco_ids.append(coco_id)
        assert len(coco_ids) == len(CLASS_NAMES), "a class name was not found in the pretrained model's names"

        new_head = self.yolo.model[-1] #Detect layer of our model
        old_head = pretrained.model[-1] #Detect layer of the coco model
        for new_branch, old_branch in zip(new_head.cv3, old_head.cv3): #one classification branch per scale
            new_conv = new_branch[-1] #last layer of the branch, gives one score per class
            old_conv = old_branch[-1]
            assert new_conv.in_channels == old_conv.in_channels, "the classification branches have different widths"
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight[coco_ids]) #keep only the rows of our classes, so the frozen head already knows them
                new_conv.bias.copy_(old_conv.bias[coco_ids])

        names = {} #class number -> class name
        for i, name in enumerate(CLASS_NAMES):
            names[i] = name
        self.yolo.names = names

        self.yolo.args = get_cfg() #default loss gains (box, cls, dfl), the ultralytics loss reads them from here

        for param in self.yolo.parameters():
            param.requires_grad_(False) #yolo is frozen, only the preproc is trained
        self.yolo.eval() #see train() below

    def train(self, mode=True):
        super().train(mode) #the preproc follows train / eval as usual
        self.yolo.eval() #yolo always stays in eval mode, otherwise its batchnorm layers would keep updating their saved statistics
        return self

    def enhance(self, images):
        #images shape: (B, 3, H, W)
        enhanced, gates = self.preproc(images) #shapes: (B, 3, H, W) and (B, 5, H, W)
        return enhanced, gates

    def compute_loss(self, images, batch):
        #batch: dict with batch_idx, cls, bboxes (normalised xywh), see make_yolo_batch
        enhanced, gates = self.enhance(images)
        batch["img"] = enhanced #the ultralytics loss runs the detector on batch["img"], so the gradient reaches the filters
        loss, loss_items = self.yolo.loss(batch) #loss is summed over the batch, loss_items are the box, cls and dfl parts
        loss = loss.sum() / images.shape[0] #average per image
        return loss

    @torch.no_grad()
    def detect(self, images, conf_thres, iou_thres, max_det):
        enhanced, gates = self.enhance(images)
        preds = self.yolo(enhanced) #in eval mode, the raw predictions that ultralytics also feeds to nms
        detections = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
        return detections #one (n, 6) tensor per image: x1, y1, x2, y2, confidence, class

    def forward(self, images):
        enhanced, gates = self.enhance(images)
        return self.yolo(enhanced)