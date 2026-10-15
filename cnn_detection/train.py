from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    CLASS_MAP_PATH,
    COORDINATE_WEIGHT,
    EPOCHS,
    FLIP_PROBABILITY,
    GAMMA_RANGE,
    GRID_SIZE,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    LEARNING_RATE,
    MAX_SHIFT,
    MIRROR_PERMUTATION,
    MODEL_DIRECTORY,
    MODEL_PATH,
    NUM_SITES,
    NUM_STATES,
    RANDOM_SEED,
    SCALE_RANGE,
    SCORE_THRESHOLD,
    SITES,
    SPLIT_ROOT,
    STATE_WEIGHT,
    STATES,
    VALUES_PER_SITE,
    WEIGHT_DECAY,
)
from .datasets.dataset import AxialDetectionDataset
from .datasets.transforms import (
    DetectionCompose,
    DetectionRandomAffine,
    DetectionRandomGamma,
    DetectionResize,
    DetectionToTensor,
)
from .detector import LeNetDetector
from .eval.metrics import evaluate
from .grid import EncodeGridTarget, GridRandomHorizontalFlip, load_class_slots
from .losses.losses import DetectionLoss

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA PIPELINE
# ============================================================

def create_loader(
    split_name: str,
    shuffle: bool,
    class_slots: list[tuple[int, int]],
    augment: bool = False,
) -> DataLoader:
    operations = [DetectionResize(IMAGE_SIZE)]

    # Augmentation is for training only
    if augment:
        operations += [
            DetectionRandomAffine(MAX_SHIFT, SCALE_RANGE),
            DetectionRandomGamma(GAMMA_RANGE),
        ]

    operations += [
        DetectionToTensor(),
        EncodeGridTarget(
            class_slots=class_slots,
            num_sites=NUM_SITES,
            grid_size=GRID_SIZE,
        ),
    ]

    if augment:
        operations.append(GridRandomHorizontalFlip(MIRROR_PERMUTATION, FLIP_PROBABILITY))

    transform = DetectionCompose(operations)

    dataset = AxialDetectionDataset(
        csv_path=SPLIT_ROOT / f"detection_{split_name}.csv",
        class_map_path=CLASS_MAP_PATH,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def inspect_labels(
    dataset: AxialDetectionDataset,
    class_slots: list[tuple[int, int]],
) -> torch.Tensor:
    """Print label statistics and return class weights for the state loss."""
    state_counts = torch.zeros(NUM_STATES)
    most_per_site = 0

    for row in dataset.rows:
        sites_in_image: Counter[int] = Counter()

        for obj in json.loads(row["objects_json"]):
            site, state = class_slots[dataset.class_map[obj["name"]] - 1]
            sites_in_image[site] += 1
            state_counts[state] += 1

        most_per_site = max(most_per_site, max(sites_in_image.values()))

    print(f"Training images: {len(dataset)}")
    print(
        "Objects per state: "
        + ", ".join(f"{name}={int(count)}" for name, count in zip(STATES, state_counts))
    )
    print(f"Most objects of one site in one image: {most_per_site}")

    if most_per_site > 1:
        raise ValueError(
            "An image has more than one object for the same site. "
            "The one-slot-per-site design cannot represent that."
        )

    # Balanced weights: rare states count more. A state with no examples gets 0.
    weights = state_counts.sum() / (NUM_STATES * state_counts.clamp(min=1))

    return torch.where(state_counts > 0, weights, torch.zeros_like(weights))


# ============================================================
# MODEL CHECK
# ============================================================

def check_model_output(model: LeNetDetector, device: torch.device) -> None:
    dummy_images = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)

    with torch.no_grad():
        output_shape = tuple(model(dummy_images).shape)

    expected_shape = (1, NUM_SITES, VALUES_PER_SITE, GRID_SIZE, GRID_SIZE)

    if output_shape != expected_shape:
        raise ValueError(
            f"Model output {output_shape} does not match {expected_shape}"
        )

    print(f"Model output shape: {output_shape}")


# ============================================================
# REPORTING
# ============================================================

def format_state_recall(recall: list[float]) -> str:
    return " ".join(f"{name}={value:.1%}" for name, value in zip(STATES, recall))


def print_report(title: str, report: dict) -> None:
    site = report["site"]

    print(title)
    print(f"  {'site':<5} {'GT':>5} {'found':>7} {'state':>7} {'joint':>7} {'IoU':>6} {'FP':>5}")

    for index, name in enumerate(SITES):
        print(
            f"  {name:<5} {int(site['gt'][index]):>5} "
            f"{site['found_rate'][index]:>7.1%} "
            f"{site['state_accuracy'][index]:>7.1%} "
            f"{site['joint_accuracy'][index]:>7.1%} "
            f"{site['mean_iou'][index]:>6.3f} "
            f"{int(site['false_positive'][index]):>5}"
        )

    print(f"  state recall: {format_state_recall(report['state']['recall'])}")
    print(f"  balanced joint accuracy: {report['balanced_joint']:.4f}")


# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: DetectionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    totals: dict[str, float] = {}
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        losses = loss_function(model(images), targets)

        losses["total"].backward()
        optimizer.step()

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + value.item() * images.size(0)

        total_samples += images.size(0)

    return {name: value / total_samples for name, value in totals.items()}


def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    return evaluate(
        model=model,
        loader=loader,
        device=device,
        image_size=IMAGE_SIZE,
        num_sites=NUM_SITES,
        num_states=NUM_STATES,
        score_threshold=SCORE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_random_seed(RANDOM_SEED)
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    class_slots = load_class_slots(CLASS_MAP_PATH, SITES, STATES)

    training_loader = create_loader("train", shuffle=True, class_slots=class_slots, augment=True)
    validation_loader = create_loader("val", shuffle=False, class_slots=class_slots)
    test_loader = create_loader("test", shuffle=False, class_slots=class_slots)

    state_weights = inspect_labels(training_loader.dataset, class_slots)

    model = LeNetDetector(
        num_sites=NUM_SITES,
        num_states=NUM_STATES,
    ).to(device)
    check_model_output(model, device)

    loss_function = DetectionLoss(
        state_weights=state_weights,
        coordinate_weight=COORDINATE_WEIGHT,
        state_weight=STATE_WEIGHT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Learning rate goes from LEARNING_RATE down to 0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_balanced_joint = -1.0

    for epoch in range(1, EPOCHS + 1):
        training_losses = train_one_epoch(
            model=model,
            loader=training_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        scheduler.step()

        report = run_evaluation(model, validation_loader, device)
        balanced_joint = report["balanced_joint"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={training_losses['total']:.4f} "
            f"(object={training_losses['object']:.4f} "
            f"offset={training_losses['offset']:.4f} "
            f"size={training_losses['size']:.4f} "
            f"state={training_losses['state']:.4f}) | "
            f"validation found={report['found_rate']:.1%} | "
            f"state recall: {format_state_recall(report['state']['recall'])} | "
            f"balanced={balanced_joint:.4f}"
        )

        if balanced_joint > best_balanced_joint:
            best_balanced_joint = balanced_joint

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": {
                        "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE],
                        "grid_size": GRID_SIZE,
                        "sites": list(SITES),
                        "states": list(STATES),
                        "values_per_site": VALUES_PER_SITE,
                    },
                },
                MODEL_PATH,
            )

            print(f"Saved improved model to: {MODEL_PATH}")

    print()
    print("========================================")
    print("Training complete")
    print(f"Best validation balanced joint accuracy: {best_balanced_joint:.4f}")
    print(f"Model: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print_report("Validation (best model)", run_evaluation(model, validation_loader, device))
    print_report("Test (best model)", run_evaluation(model, test_loader, device))
    print("========================================")


if __name__ == "__main__":
    main()