"""Detection transforms that keep image and bounding boxes synchronized."""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image


class DetectionToTensor:
    """Convert grayscale PIL image to float tensor [1,H,W], normalized per image."""

    def __call__(self, image, target):
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
        return tensor, target


class DetectionResize:
    """Resize image and scale xyxy boxes. `size` is int or (width, height)."""

    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, image, target):
        old_w, old_h = image.size
        new_w, new_h = self.size
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        target = dict(target)
        boxes = target["boxes"].clone()
        if boxes.numel():
            boxes[:, [0, 2]] *= new_w / old_w
            boxes[:, [1, 3]] *= new_h / old_h
        target["boxes"] = boxes
        if "area" in target:
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return image, target


class DetectionRandomAffine:
    """Random shift and zoom around the image center; the boxes follow the image."""

    def __init__(self, max_shift=10, scale_range=(0.9, 1.1)):
        self.max_shift = max_shift
        self.scale_range = scale_range

    def __call__(self, image, target):
        width, height = image.size
        center_x, center_y = width / 2, height / 2

        scale = random.uniform(*self.scale_range)
        shift_x = random.uniform(-self.max_shift, self.max_shift)
        shift_y = random.uniform(-self.max_shift, self.max_shift)

        # PIL maps every output pixel back to an input pixel:
        #     x_in = (x_out - center - shift) / scale + center
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (
                1 / scale, 0, center_x - (center_x + shift_x) / scale,
                0, 1 / scale, center_y - (center_y + shift_y) / scale,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        target = dict(target)
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - center_x) * scale + center_x + shift_x).clamp(0, width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - center_y) * scale + center_y + shift_y).clamp(0, height)
        target["boxes"] = boxes
        if "area" in target:
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return image, target


class DetectionRandomGamma:
    """
    Random gamma on the intensities.

    A plain brightness or contrast change would have no effect, because
    DetectionToTensor normalizes every image by its own mean and std.
    """

    def __init__(self, gamma_range=(0.8, 1.25)):
        self.gamma_range = gamma_range

    def __call__(self, image, target):
        gamma = random.uniform(*self.gamma_range)
        table = [round(255 * (i / 255) ** gamma) for i in range(256)]
        return image.point(table), target


class DetectionRandomHorizontalFlip:
    """Horizontal flip with correct box updates.

    Disabled by default: laterality-specific classes (e.g. left/right foraminal
    stenosis) can become semantically incorrect unless their labels are swapped.
    For training, use GridRandomHorizontalFlip in grid.py, which swaps them.
    """

    def __init__(self, p=0.0):
        self.p = float(p)

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        width, _ = image.size
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target = dict(target)
        boxes = target["boxes"].clone()
        if boxes.numel():
            old_xmin = boxes[:, 0].clone()
            old_xmax = boxes[:, 2].clone()
            boxes[:, 0] = width - old_xmax
            boxes[:, 2] = width - old_xmin
        target["boxes"] = boxes
        if "area" in target:
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return image, target


class DetectionCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def build_default_transform(resize=None):
    """Default transform; resize=None preserves native resolution."""
    ops = []
    if resize is not None:
        ops.append(DetectionResize(resize))
    ops.append(DetectionToTensor())
    return DetectionCompose(ops)