from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    FOG_DIR, FOG_SPLIT_FILES, FOG_TEST_SPLITS, WEATHER_TRAIN_SPLITS, FOG_BETAS, FOGGY_PROB,
    DEVICE, NUM_WORKERS, DET_BATCH_SIZE,
    AUG_FOG, AUG_SENSOR, AUG_GEOM,
    AUG_CUSTOM_FOG_PROB, AUG_FOG_TAU_NEAR, AUG_FOG_TAU_FAR, AUG_FOG_GRADIENT_GAMMA, AUG_FOG_NOISE_AMOUNT,
    AUG_AIRLIGHT_RANGE, AUG_AIRLIGHT_TINT,
    AUG_BLUR_PROB, AUG_BLUR_SIGMA, AUG_NOISE_PROB, AUG_NOISE_SIGMA, AUG_JPEG_PROB, AUG_JPEG_QUALITY,
    AUG_CROP_PROB, AUG_CROP_SIDE,
    AUG_AFFINE_PROB, AUG_AFFINE_SCALE, AUG_AFFINE_ROTATE, AUG_AFFINE_SHEAR, AUG_AFFINE_TRANSLATE,
    AUG_BOX_MIN_SIDE, AUG_BOX_MIN_VISIBLE,
)

PAD_VALUE = 128 / 255.0 #grey of the letterbox bars, the new borders made by the affine are filled with the same grey


def read_annotation_file(split):
    #each line is: relative/path.jpg x1,y1,x2,y2,class x1,y1,x2,y2,class ...
    annotation_path = FOG_DIR / "annotations" / FOG_SPLIT_FILES[split] #the annotation file for this split
    lines = annotation_path.read_text(encoding="utf-8").splitlines() #one line per image

    samples = [] #one (relative path, boxes) pair per image
    for line in lines:
        parts = line.split() #first part is the path, the rest are boxes
        if len(parts) > 0: #skip blank lines
            relative_path = parts[0] #path relative to the fog_416 folder

            boxes = [] #one [x1, y1, x2, y2, class] per box
            for box_text in parts[1:]:
                values = [float(value) for value in box_text.split(",")]
                boxes.append(values)
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 5) #shape: (N, 5), the reshape keeps images with no boxes valid

            samples.append((relative_path, boxes))
    return samples


def foggy_path_for(clean_relative_path, beta):
    stem = Path(clean_relative_path).stem #for example VOC2007_000007
    return f"train/foggy/{stem}_{beta}.jpg" #the same image with fog level beta


# ---------------------------------------------------------------------------------------------
# augmentation helpers, images are float32 numpy arrays of shape (H, W, 3), rgb, values 0 to 1
# all random numbers come from torch, so the dataloader seeds every worker differently
# ---------------------------------------------------------------------------------------------

def rand_uniform(low, high):
    return low + (high - low) * torch.rand(1).item() #a random number between low and high


def rand_chance(probability):
    return torch.rand(1).item() < probability #true with this probability


def read_rgb(image_path):
    image = cv2.imread(str(image_path)) #bgr, shape: (416, 416, 3), uint8
    if image is None:
        raise FileNotFoundError(f"could not read {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #rgb
    return image.astype(np.float32) / np.float32(255.0) #values 0 to 1


def content_box(image):
    #the images are letterboxed, grey bars on two sides, this finds the part that is the real photo
    height, width = image.shape[:2]
    not_pad = np.abs(image - PAD_VALUE).max(axis=2) > 0.01 #pixels that are not bar grey
    rows = np.where(not_pad.mean(axis=1) > 0.02)[0] #rows that are mostly photo
    cols = np.where(not_pad.mean(axis=0) > 0.02)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, height, 0, width #nothing found, use the whole image
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def low_frequency_noise(height, width):
    grid_h = int(rand_uniform(3, 7)) #3 to 6 cells
    grid_w = int(rand_uniform(3, 7))
    grid = torch.rand(grid_h, grid_w).numpy().astype(np.float32) #a few random values
    noise = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC) #stretched into a smooth map
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-6) #rescaled to 0 to 1


def render_depth_fog(image):
    #I = J * t + A * (1 - t), t = exp(-tau), tau is small near the camera (bottom) and large far away (top)
    y0, y1, x0, x1 = content_box(image) #the bars are left untouched
    content_h, content_w = y1 - y0, x1 - x0

    rows = np.linspace(1.0, 0.0, content_h).astype(np.float32) #1 at the top (far), 0 at the bottom (near)
    gradient = np.power(rows, rand_uniform(*AUG_FOG_GRADIENT_GAMMA))[:, None] #shape: (h, 1)
    noise = low_frequency_noise(content_h, content_w) #shape: (h, w)
    noise_amount = rand_uniform(*AUG_FOG_NOISE_AMOUNT)
    depth = np.clip((1 - noise_amount) * gradient + noise_amount * noise, 0.0, 1.0) #stand-in for depth, 0 near, 1 far

    tau_far = rand_uniform(*AUG_FOG_TAU_FAR)
    tau_near = min(rand_uniform(*AUG_FOG_TAU_NEAR), 0.6 * tau_far) #the near field is always clearer than the far field
    tau = tau_near + (tau_far - tau_near) * depth #shape: (h, w)
    t = np.exp(-tau)[:, :, None].astype(np.float32) #transmission, shape: (h, w, 1)

    tint = np.array([rand_uniform(-AUG_AIRLIGHT_TINT, AUG_AIRLIGHT_TINT) for _ in range(3)], dtype=np.float32)
    airlight = np.clip(rand_uniform(*AUG_AIRLIGHT_RANGE) + tint, 0.0, 1.0).reshape(1, 1, 3) #the fog colour

    fogged = image.copy()
    fogged[y0:y1, x0:x1] = image[y0:y1, x0:x1] * t + airlight * (1 - t)
    return np.clip(fogged, 0.0, 1.0).astype(np.float32)


def sensor_effects(image):
    #blur, noise and jpeg compression, each with its own chance, the letterbox bars are put back exactly afterwards
    y0, y1, x0, x1 = content_box(image)
    out = image.copy()

    if rand_chance(AUG_BLUR_PROB):
        out = cv2.GaussianBlur(out, (0, 0), rand_uniform(*AUG_BLUR_SIGMA))

    if rand_chance(AUG_NOISE_PROB):
        sigma = rand_uniform(*AUG_NOISE_SIGMA)
        out = out + sigma * torch.randn(*out.shape).numpy().astype(np.float32)
    out = np.clip(out, 0.0, 1.0)

    if rand_chance(AUG_JPEG_PROB):
        quality = int(rand_uniform(*AUG_JPEG_QUALITY))
        bgr = cv2.cvtColor((out * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            out = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255.0)

    result = image.copy()
    result[y0:y1, x0:x1] = out[y0:y1, x0:x1] #bars stay exactly as they were
    return result.astype(np.float32)


def transform_boxes(boxes, matrix, height, width):
    #boxes: (N, 5) [x1, y1, x2, y2, class], matrix: 2x3 affine, the four corners of every box are moved and re-boxed
    if boxes.shape[0] == 0:
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack([
        np.stack([x1, y1], axis=1), np.stack([x2, y1], axis=1),
        np.stack([x2, y2], axis=1), np.stack([x1, y2], axis=1),
    ], axis=1) #shape: (N, 4, 2)
    moved = corners @ matrix[:, :2].T + matrix[:, 2] #shape: (N, 4, 2)

    new_x1, new_x2 = moved[..., 0].min(axis=1), moved[..., 0].max(axis=1)
    new_y1, new_y2 = moved[..., 1].min(axis=1), moved[..., 1].max(axis=1)
    full_area = (new_x2 - new_x1) * (new_y2 - new_y1) #area before cutting at the image edge

    clip_x1, clip_x2 = np.clip(new_x1, 0, width), np.clip(new_x2, 0, width)
    clip_y1, clip_y2 = np.clip(new_y1, 0, height), np.clip(new_y2, 0, height)
    clip_w, clip_h = clip_x2 - clip_x1, clip_y2 - clip_y1
    visible = (clip_w * clip_h) / (full_area + 1e-6) #fraction of the box still inside the image

    keep = (clip_w > AUG_BOX_MIN_SIDE) & (clip_h > AUG_BOX_MIN_SIDE) & (visible > AUG_BOX_MIN_VISIBLE)
    new_boxes = np.stack([clip_x1, clip_y1, clip_x2, clip_y2, boxes[:, 4]], axis=1)[keep]
    return new_boxes.astype(np.float32).reshape(-1, 5)


def warp(image, matrix):
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )


def random_crop_matrix(height, width):
    #keep a random window of the image and stretch it back to full size, so this only ever zooms in
    side = rand_uniform(*AUG_CROP_SIDE) #fraction of the side that is kept
    zoom = 1.0 / side
    x0 = rand_uniform(0.0, 1.0 - side) * width #left edge of the window
    y0 = rand_uniform(0.0, 1.0 - side) * height #top edge of the window
    return np.array([[zoom, 0.0, -x0 * zoom], [0.0, zoom, -y0 * zoom]], dtype=np.float64)


def random_affine_matrix(height, width):
    scale = rand_uniform(*AUG_AFFINE_SCALE) #below 1 zooms out
    angle = np.deg2rad(rand_uniform(-AUG_AFFINE_ROTATE, AUG_AFFINE_ROTATE))
    shear = np.deg2rad(rand_uniform(-AUG_AFFINE_SHEAR, AUG_AFFINE_SHEAR))
    shift_x = rand_uniform(-AUG_AFFINE_TRANSLATE, AUG_AFFINE_TRANSLATE) * width
    shift_y = rand_uniform(-AUG_AFFINE_TRANSLATE, AUG_AFFINE_TRANSLATE) * height

    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shear_matrix = np.array([[1.0, np.tan(shear)], [0.0, 1.0]])
    linear = scale * rotation @ shear_matrix #shape: (2, 2)

    center = np.array([width / 2.0, height / 2.0])
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = center + np.array([shift_x, shift_y]) - linear @ center #rotate, scale and shear around the centre
    return matrix


def geometric_augment(image, boxes):
    height, width = image.shape[:2]
    steps = []
    if rand_chance(AUG_CROP_PROB):
        steps.append(random_crop_matrix)
    if rand_chance(AUG_AFFINE_PROB):
        steps.append(random_affine_matrix)

    for make_matrix in steps:
        matrix = make_matrix(height, width)
        new_boxes = transform_boxes(boxes, matrix, height, width)
        if boxes.shape[0] > 0 and new_boxes.shape[0] == 0:
            continue #this step would remove every box, skip it
        image = warp(image, matrix)
        boxes = new_boxes
    return image, boxes


class FogDataset(Dataset): #one split of fog_416 (or of the combined dataset), boxes are [x1, y1, x2, y2, class] in pixels of the 416 x 416 image

    def __init__(self, split, use_flip=False, foggy_prob=FOGGY_PROB, use_augment=None):
        self.split = split #train, test_foggy, test_clean, rtts, or test_<weather> with the combined dataset
        self.use_flip = use_flip #random left-right flip, only for training
        self.foggy_prob = foggy_prob #chance of using a foggy copy, only used for train
        self.use_augment = (split == "train") if use_augment is None else use_augment #augmentation never touches the test splits
        self.samples = read_annotation_file(split) #list of (relative path, boxes)
        self.can_fog = [True] * len(self.samples) #the fog_416 clean images have foggy copies and can be fogged

        if split == "train":
            #combined dataset: the real bad weather images join the training set, WEATHER_TRAIN_SPLITS is empty with fog_416
            #they are already bad weather, so they are never swapped for a foggy copy, never fogged and get no sensor effects
            #flip and the geometric augmentation still apply to them
            for weather_split in WEATHER_TRAIN_SPLITS:
                weather_samples = read_annotation_file(weather_split)
                self.samples += weather_samples
                self.can_fog += [False] * len(weather_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative_path, boxes = self.samples[index]
        boxes = boxes.copy() #so the transforms do not change the stored boxes

        is_foggy = False #is this a foggy sample
        use_custom_fog = False #render our own fog on the clean image
        if self.split == "train" and self.can_fog[index]:
            if rand_chance(self.foggy_prob): #2/3 of the time use a foggy copy, 1/3 keep the clean one
                is_foggy = True
                if self.use_augment and AUG_FOG and rand_chance(AUG_CUSTOM_FOG_PROB):
                    use_custom_fog = True #keep the clean path, the fog is rendered below
                else:
                    beta_index = torch.randint(len(FOG_BETAS), (1,)).item() #pick one of the 10 fog levels
                    relative_path = foggy_path_for(relative_path, FOG_BETAS[beta_index])

        image = read_rgb(FOG_DIR / relative_path) #shape: (416, 416, 3), float32, 0 to 1

        if self.use_augment and self.split == "train":
            if use_custom_fog:
                image = render_depth_fog(image) #depth-dependent fog with a random colour
            if is_foggy and AUG_SENSOR:
                image = sensor_effects(image) #blur, noise, jpeg, after the fog
            if AUG_GEOM:
                image, boxes = geometric_augment(image, boxes) #random crop and affine, boxes move with the image

        if self.use_flip and rand_chance(0.5):
            width = image.shape[1] #image width in pixels
            image = np.ascontiguousarray(image[:, ::-1]) #mirror left to right

            new_x1 = width - boxes[:, 2] #the old right edge becomes the new left edge
            new_x2 = width - boxes[:, 0] #the old left edge becomes the new right edge
            boxes[:, 0] = new_x1
            boxes[:, 2] = new_x2

        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))) #shape: (3, 416, 416)
        return image, torch.from_numpy(boxes)


def collate_fn(samples):
    #samples: list of (image, boxes), the number of boxes differs between images so they cannot be stacked
    images = []
    boxes_list = []
    for image, boxes in samples:
        images.append(image)
        boxes_list.append(boxes)

    images = torch.stack(images, dim=0) #shape: (B, 3, 416, 416)
    return images, boxes_list #boxes_list has one (N, 5) tensor per image


def make_train_loader(seed, batch_size=DET_BATCH_SIZE):
    generator = torch.Generator().manual_seed(seed) #same shuffling for the same seed
    dataset = FogDataset("train", use_flip=True)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), collate_fn=collate_fn,
    ) #drop_last so the last batch is never a single image


def make_test_loader(split, batch_size=DET_BATCH_SIZE):
    dataset = FogDataset(split)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), collate_fn=collate_fn,
    )


def make_test_loaders(batch_size=DET_BATCH_SIZE):
    test_loaders = {} #split name -> loader
    for split in FOG_TEST_SPLITS:
        test_loaders[split] = make_test_loader(split, batch_size)
    return test_loaders