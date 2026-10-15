"""Confusion matrix of the best checkpoint.

Run from the project root:

    python -m src.cnn_detection.confusion --split test

Rows are the true state, columns are the predicted state. The last column,
"missed", counts sites the model did not find (score below the threshold
or box IoU below the threshold), so it is a localization error, not a
state error. The diagonal share of each row is the recall printed by train.py.
"""
from __future__ import annotations

import argparse

import torch

from .config import (
    CLASS_MAP_PATH,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    MODEL_PATH,
    NUM_SITES,
    NUM_STATES,
    SCORE_THRESHOLD,
    SITES,
    STATES,
)
from .detector import LeNetDetector
from .eval.metrics import box_iou
from .grid import decode_predictions, decode_targets, load_class_slots
from .train import create_loader


@torch.no_grad()
def count_confusion(
    model: LeNetDetector,
    loader,
    device: torch.device,
) -> torch.Tensor:
    """Counts with shape [sites, true state, predicted state + missed]."""
    model.eval()

    counts = torch.zeros(NUM_SITES, NUM_STATES, NUM_STATES + 1, dtype=torch.long)

    for images, targets in loader:
        detections = decode_predictions(model(images.to(device)), IMAGE_SIZE)
        truth = decode_targets(targets.to(device), IMAGE_SIZE)

        iou = box_iou(detections.boxes, truth.boxes)
        found = truth.present & (detections.scores >= SCORE_THRESHOLD) & (iou >= IOU_THRESHOLD)

        # Predicted state, or the last column when the site was not found
        predicted = torch.where(
            found,
            detections.states,
            torch.full_like(detections.states, NUM_STATES),
        )

        site = torch.arange(NUM_SITES, device=device).expand_as(truth.states)
        keep = truth.present

        index = (site[keep] * NUM_STATES + truth.states[keep]) * (NUM_STATES + 1) + predicted[keep]
        counts += torch.bincount(index, minlength=counts.numel()).view_as(counts).cpu()

    return counts


def print_matrix(title: str, matrix: torch.Tensor) -> None:
    corner = "true / predicted"

    print(title)
    print(f"  {corner:<18}" + "".join(f"{name:>16}" for name in (*STATES, "missed")))

    for name, row in zip(STATES, matrix.tolist()):
        total = max(sum(row), 1)
        cells = "".join(f"{f'{count} ({count / total:.0%})':>16}" for count in row)
        print(f"  {name:<18}{cells}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_slots = load_class_slots(CLASS_MAP_PATH, SITES, STATES)
    loader = create_loader(args.split, shuffle=False, class_slots=class_slots)

    model = LeNetDetector(num_sites=NUM_SITES, num_states=NUM_STATES).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    counts = count_confusion(model, loader, device)

    print_matrix(f"All sites ({args.split})", counts.sum(dim=0))

    for index, name in enumerate(SITES):
        print_matrix(name, counts[index])


if __name__ == "__main__":
    main()