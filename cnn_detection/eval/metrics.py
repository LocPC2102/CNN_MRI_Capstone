from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..grid import decode_predictions, decode_targets


def box_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """Element-wise IoU of xyxy boxes with shape [..., 4]."""
    top_left = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bottom_right = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area_a = (boxes_a[..., 2:] - boxes_a[..., :2]).clamp(min=0).prod(dim=-1)
    area_b = (boxes_b[..., 2:] - boxes_b[..., :2]).clamp(min=0).prod(dim=-1)

    return intersection / (area_a + area_b - intersection).clamp(min=1e-6)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    num_sites: int,
    num_states: int,
    score_threshold: float,
    iou_threshold: float,
) -> dict:
    """
    For every site of every image:

        found    the site is annotated, predicted (score >= threshold)
                 and the box has IoU >= iou_threshold
        correct  found, and the predicted state equals the true state

    Returns lists per site and per state, plus `balanced_joint`:
    the mean over states of (correct / annotated), which is not fooled
    by a model that always answers "normal".
    """
    model.eval()

    site_gt = torch.zeros(num_sites, device=device)
    site_found = torch.zeros(num_sites, device=device)
    site_correct = torch.zeros(num_sites, device=device)
    site_false_positive = torch.zeros(num_sites, device=device)
    site_iou_sum = torch.zeros(num_sites, device=device)

    state_gt = torch.zeros(num_states, device=device)
    state_correct = torch.zeros(num_states, device=device)

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        detections = decode_predictions(model(images), image_size)
        truth = decode_targets(targets, image_size)

        iou = box_iou(detections.boxes, truth.boxes)
        predicted = detections.scores >= score_threshold

        found = truth.present & predicted & (iou >= iou_threshold)
        correct = found & (detections.states == truth.states)

        site_gt += truth.present.sum(dim=0)
        site_found += found.sum(dim=0)
        site_correct += correct.sum(dim=0)
        site_false_positive += (~truth.present & predicted).sum(dim=0)
        site_iou_sum += (iou * found).sum(dim=0)

        for state in range(num_states):
            is_state = truth.present & (truth.states == state)
            state_gt[state] += is_state.sum()
            state_correct[state] += (correct & is_state).sum()

    state_recall = state_correct / state_gt.clamp(min=1)

    return {
        "site": {
            "gt": site_gt.tolist(),
            "found_rate": (site_found / site_gt.clamp(min=1)).tolist(),
            "state_accuracy": (site_correct / site_found.clamp(min=1)).tolist(),
            "joint_accuracy": (site_correct / site_gt.clamp(min=1)).tolist(),
            "mean_iou": (site_iou_sum / site_found.clamp(min=1)).tolist(),
            "false_positive": site_false_positive.tolist(),
        },
        "state": {
            "gt": state_gt.tolist(),
            "recall": state_recall.tolist(),
        },
        "found_rate": (site_found.sum() / site_gt.sum().clamp(min=1)).item(),
        "balanced_joint": state_recall[state_gt > 0].mean().item(),
    }