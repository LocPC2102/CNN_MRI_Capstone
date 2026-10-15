"""Split annotated detection samples by patient to prevent leakage.

Run from project root:
    python src/cnn_detector_axial/datasets/split_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

VALID_TRAIN_STATUSES = {"annotated"}


def split(manifest: Path, out_dir: Path, ratios=(0.70, 0.15, 0.15), seed=42):
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must be three non-negative values summing to 1")

    with manifest.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    if not fields:
        raise ValueError(f"Manifest has no header: {manifest}")

    # Only valid, non-empty annotated samples enter the default detection split.
    # Unannotated images are not negative samples and are excluded from training.
    trainable = [
        r for r in rows
        if r.get("annotation_status") in VALID_TRAIN_STATUSES and r.get("image_path")
    ]
    if not trainable:
        raise ValueError(
            "No annotated samples found. Run audit_dataset.py and inspect the manifest."
        )

    by_patient = defaultdict(list)
    for row in trainable:
        by_patient[row["patient_id"]].append(row)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    n = len(patients)
    if n < 3:
        raise ValueError(
            f"Only {n} annotated patients; cannot create meaningful train/val/test split."
        )

    n_train = max(1, int(ratios[0] * n))
    n_val = max(1, int(ratios[1] * n))
    # Keep at least one patient in test.
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    groups = {
        "train": patients[:n_train],
        "val": patients[n_train:n_train + n_val],
        "test": patients[n_train + n_val:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    split_patient_sets = []
    for name, patient_ids in groups.items():
        patient_set = set(patient_ids)
        split_rows = [r for r in trainable if r["patient_id"] in patient_set]
        with (out_dir / f"detection_{name}.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows)
        total_written += len(split_rows)
        split_patient_sets.append(patient_set)
        print(
            f"{name:5s}: {len(patient_ids):4d} patients, "
            f"{len(split_rows):5d} annotated images"
        )

    assert not (split_patient_sets[0] & split_patient_sets[1])
    assert not (split_patient_sets[0] & split_patient_sets[2])
    assert not (split_patient_sets[1] & split_patient_sets[2])
    assert total_written == len(trainable)
    print(f"OK: no patient overlap; {total_written} annotated images; seed={seed}")
    print(f"Unannotated/invalid rows excluded from splits: {len(rows) - len(trainable)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="DATA/metadata/detection_manifest.csv")
    parser.add_argument("--out-dir", default="DATA/processed/detection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    split(
        Path(args.manifest),
        Path(args.out_dir),
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )