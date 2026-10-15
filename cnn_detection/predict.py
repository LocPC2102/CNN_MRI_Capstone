"""Compare detector predictions with the ground truth, one picture per slice.

Run from the project root:

    python -m src.cnn_detection.predict --split test --count 6
    python -m src.cnn_detection.predict --image DATA/raw/.../0001/IM000003.png

An XML with the same name next to the image is used as the ground truth.
Every site keeps the same number and colour in all panels (see SITES).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from .config import (
    IMAGE_SIZE,
    MODEL_PATH,
    NUM_SITES,
    NUM_STATES,
    SCORE_THRESHOLD,
    SITES,
    SPLIT_ROOT,
    STATES,
)
from .datasets.audit_dataset import parse_xml
from .datasets.transforms import DetectionCompose, DetectionResize, DetectionToTensor
from .detector import LeNetDetector
from .eval.metrics import box_iou
from .grid import decode_predictions

# One colour per site, in the order of SITES
SITE_COLORS = ("#F5A623", "#34A853", "#4285F4", "#EA4335", "#A142F4")
GOOD, BAD, NEUTRAL = "#1e8e3e", "#d93025", "#555555"

# Same preprocessing as training
TRANSFORM = DetectionCompose([DetectionResize(IMAGE_SIZE), DetectionToTensor()])


# ============================================================
# MODEL
# ============================================================

def load_model(device: torch.device) -> LeNetDetector:
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = LeNetDetector(num_sites=NUM_SITES, num_states=NUM_STATES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model.eval()


@torch.no_grad()
def predict(
    model: LeNetDetector,
    image: Image.Image,
    device: torch.device,
    threshold: float,
) -> dict[int, dict]:
    """Detections above the threshold, keyed by site index, boxes in original pixels."""
    width, height = image.size

    tensor, _ = TRANSFORM(image, {"boxes": torch.zeros(0, 4)})
    detections = decode_predictions(model(tensor[None].to(device)), IMAGE_SIZE)

    to_original = torch.tensor([width, height, width, height]) / IMAGE_SIZE

    found = {}
    for site in range(NUM_SITES):
        score = detections.scores[0, site].item()
        if score < threshold:
            continue

        state = detections.states[0, site].item()
        found[site] = {
            "box": (detections.boxes[0, site].cpu() * to_original).tolist(),
            "state": state,
            "score": score,
            "state_probability": detections.state_probabilities[0, site, state].item(),
        }

    return found


# ============================================================
# GROUND TRUTH
# ============================================================

def truth_from_objects(objects: list[dict]) -> dict[int, dict]:
    """Ground truth keyed by site index, from manifest or XML objects."""
    truth = {}

    for obj in objects:
        site, _, state = obj["name"].partition("_")

        if site in SITES and state in STATES:
            truth[SITES.index(site)] = {
                "box": obj["bbox"],
                "state": STATES.index(state),
            }

    return truth


def collect_samples(args: argparse.Namespace):
    """Yield (image path, ground-truth objects or None when unknown)."""
    if args.image:
        for path in args.image:
            yield path, None
        return

    with open(SPLIT_ROOT / f"detection_{args.split}.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for row in random.Random(args.seed).sample(rows, min(args.count, len(rows))):
        yield Path(row["image_path"]), json.loads(row["objects_json"])


# ============================================================
# DRAWING
# ============================================================

def zoom_limits(boxes: list[list[float]], width: int, height: int):
    """Square window around all boxes, so the small structures are easy to see."""
    if not boxes:
        return (0, width), (height, 0)

    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[2] for b in boxes)
    y_max = max(b[3] for b in boxes)

    half = min(max(x_max - x_min, y_max - y_min) / 2 + 40, width / 2, height / 2)
    center_x = min(max((x_min + x_max) / 2, half), width - half)
    center_y = min(max((y_min + y_max) / 2, half), height - half)

    return (center_x - half, center_x + half), (center_y + half, center_y - half)


def draw_panel(ax, image, title, boxes_by_site, xlim, ylim) -> None:
    ax.imshow(image, cmap="gray")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    for site, (x_min, y_min, x_max, y_max) in boxes_by_site.items():
        color = SITE_COLORS[site]

        ax.add_patch(Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=False, edgecolor=color, linewidth=1.8,
        ))

        # Numbered circle just above the box
        ax.annotate(
            str(site + 1),
            xy=((x_min + x_max) / 2, y_min),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.25", facecolor=color, edgecolor="white", linewidth=0.8),
        )


def describe(site: int, found: dict, truth: dict | None) -> tuple[str, list[tuple[str, str]]]:
    """Title and (text, colour) lines for one site in the details panel."""
    gt = (truth or {}).get(site)

    if site not in found:
        return f"{SITES[site]} — not detected", [(f"GT: {STATES[gt['state']]}", BAD)]

    pred = found[site]
    info = f"score {pred['score']:.2f} | state {pred['state_probability']:.2f}"
    lines = [(info, NEUTRAL)]

    if gt:
        iou = box_iou(torch.tensor(pred["box"]), torch.tensor(gt["box"])).item()
        same = gt["state"] == pred["state"]
        lines[0] = (f"{info} | IoU {iou:.2f}", NEUTRAL)
        lines.append((f"GT: {STATES[gt['state']]} {'✓' if same else '✗'}", GOOD if same else BAD))
    elif truth is not None:
        lines.append(("GT: none", NEUTRAL))

    return f"{SITES[site]} — {STATES[pred['state']]}", lines


def draw_details(ax, found: dict, truth: dict | None) -> None:
    ax.axis("off")
    put = partial(ax.text, transform=ax.transAxes, va="center")

    put(0.02, 0.95, f"Predictions: {len(found)}", fontsize=13)

    sites = [s for s in range(NUM_SITES) if s in found or s in (truth or {})]
    if not sites:
        put(0.02, 0.85, "No detections above threshold.", fontsize=9, color=NEUTRAL)

    for row, site in enumerate(sites):
        y = 0.84 - row * 0.165
        title, lines = describe(site, found, truth)

        put(
            0.06, y, str(site + 1),
            ha="center", fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor=SITE_COLORS[site], edgecolor="none"),
        )
        put(0.14, y + 0.035, title, fontsize=10)

        for offset, (text, color) in enumerate(lines):
            put(0.14, y - 0.01 - offset * 0.04, text, fontsize=8, color=color)


def render(
    path: Path,
    image: Image.Image,
    found: dict,
    truth: dict | None,
    zoom: bool,
    output_path: Path,
) -> None:
    width, height = image.size

    all_boxes = [p["box"] for p in found.values()] + [t["box"] for t in (truth or {}).values()]
    xlim, ylim = zoom_limits(all_boxes, width, height) if zoom else ((0, width), (height, 0))

    panels = [("Input MRI", {})]
    if truth is not None:
        panels.append(("Ground truth", {s: t["box"] for s, t in truth.items()}))
    panels.append(("AI prediction", {s: p["box"] for s, p in found.items()}))

    columns = len(panels) + 1
    fig, axes = plt.subplots(
        1, columns,
        figsize=(4.4 * columns, 4.8),
        gridspec_kw={"width_ratios": [1] * len(panels) + [1.2]},
        layout="constrained",
    )

    # zip stops at the shorter list, so the last axis is left for the details
    for ax, (title, boxes_by_site) in zip(axes, panels):
        draw_panel(ax, image, title, boxes_by_site, xlim, ylim)

    draw_details(axes[-1], found, truth)

    fig.suptitle(f"{path.parent.name}/{path.name}", x=0.01, ha="left", fontsize=10)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", nargs="+", type=Path, help="image path(s) to predict")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=6, help="random images taken from --split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--out-dir", type=Path, default=Path("src/cnn_detection/predictions"))
    parser.add_argument("--full", action="store_true", help="show the whole slice, no zoom")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path, objects in collect_samples(args):
        image = Image.open(path).convert("L")

        xml_path = path.with_suffix(".xml")
        if objects is None and xml_path.exists():
            objects, _ = parse_xml(xml_path, *image.size)

        truth = None if objects is None else truth_from_objects(objects)
        found = predict(model, image, device, args.threshold)

        output_path = args.out_dir / f"{path.parent.name}_{path.stem}.png"
        render(path, image, found, truth, not args.full, output_path)

        print(f"{path} -> {output_path}")


if __name__ == "__main__":
    main()