"""Convert boxes to the detector grid and back.

Grid layout per site (one slot for each of CCS, LLS, RLS, LFS, RFS):

    channel 0  objectness   target: gaussian heatmap, exactly 1.0 at the center cell
    channel 1  offset_x     target: box center inside its cell, in [0, 1)
    channel 2  offset_y
    channel 3  log_width    target: log(box width  / cell size)
    channel 4  log_height   target: log(box height / cell size)
    channel 5  state        target: state index at the center cell, -1 elsewhere
                            prediction: first of num_states logits

Targets have shape [num_sites, 6, grid, grid];
predictions have shape [num_sites, 5 + num_states, grid, grid].
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import NamedTuple

import torch

OBJECT = 0
OFFSET_X = 1
OFFSET_Y = 2
LOG_WIDTH = 3
LOG_HEIGHT = 4
STATE = 5

# exp(4) = 55 cells; keeps decoded boxes finite
MAX_LOG_SIZE = 4.0


# ============================================================
# CLASS NAMES -> (SITE, STATE)
# ============================================================

def load_class_slots(
    class_map_path: Path,
    sites: tuple[str, ...],
    states: tuple[str, ...],
) -> list[tuple[int, int]]:
    """Return (site index, state index) for every class id; entry 0 is class id 1."""
    class_map = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
    slots: list[tuple[int, int]] = [(0, 0)] * len(class_map)

    for name, class_id in class_map.items():
        site, _, state = name.partition("_")

        if site not in sites or state not in states:
            raise ValueError(
                f"Class '{name}' does not fit SITES / STATES in config.py"
            )

        slots[class_id - 1] = (sites.index(site), states.index(state))

    return slots


# ============================================================
# ENCODE: boxes -> grid target
# ============================================================

class EncodeGridTarget:
    """
    Dataset transform. Place it after DetectionResize and DetectionToTensor.

    Input:
        image  [1, H, W] and a target dict with xyxy "boxes" and class "labels"

    Output:
        image, grid target [num_sites, 6, grid, grid]
    """

    def __init__(
        self,
        class_slots: list[tuple[int, int]],
        num_sites: int,
        grid_size: int,
        sigma: float = 1.0,
    ) -> None:
        self.class_slots = class_slots
        self.num_sites = num_sites
        self.grid_size = grid_size
        self.sigma = sigma

        rows, columns = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        )
        self.rows = rows
        self.columns = columns

    def __call__(
        self,
        image: torch.Tensor,
        target: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        cell_width = width / self.grid_size
        cell_height = height / self.grid_size

        grid = torch.zeros(self.num_sites, 6, self.grid_size, self.grid_size)
        grid[:, STATE] = -1.0

        for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
            site, state = self.class_slots[label - 1]
            xmin, ymin, xmax, ymax = box

            center_x = (xmin + xmax) / 2 / cell_width
            center_y = (ymin + ymax) / 2 / cell_height
            column = min(int(center_x), self.grid_size - 1)
            row = min(int(center_y), self.grid_size - 1)

            # Soft heatmap around the center; the center cell itself is exactly 1.
            distance = (
                (self.columns + 0.5 - center_x) ** 2
                + (self.rows + 0.5 - center_y) ** 2
            )
            heat = torch.exp(-distance / (2 * self.sigma ** 2))
            grid[site, OBJECT] = torch.maximum(grid[site, OBJECT], heat)

            grid[site, OBJECT, row, column] = 1.0
            grid[site, OFFSET_X, row, column] = center_x - column
            grid[site, OFFSET_Y, row, column] = center_y - row
            grid[site, LOG_WIDTH, row, column] = math.log(max((xmax - xmin) / cell_width, 1e-3))
            grid[site, LOG_HEIGHT, row, column] = math.log(max((ymax - ymin) / cell_height, 1e-3))
            grid[site, STATE, row, column] = state

        return image, grid


class GridRandomHorizontalFlip:
    """
    Dataset transform. Place it after EncodeGridTarget.

    Mirrors the image and the grid target together. Left and right sites
    trade places (for example LLS and RLS), so the labels stay correct.
    `site_permutation[i]` is the site that takes the place of site i.
    """

    def __init__(
        self,
        site_permutation: list[int],
        p: float = 0.5,
    ) -> None:
        self.site_permutation = site_permutation
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() >= self.p:
            return image, grid

        grid = grid[self.site_permutation].flip(-1)

        # The center moves to the mirrored cell, so its place inside the cell mirrors too
        center = grid[:, OBJECT] == 1.0
        grid[:, OFFSET_X] = torch.where(center, 1.0 - grid[:, OFFSET_X], grid[:, OFFSET_X])

        return image.flip(-1), grid


# ============================================================
# DECODE: grid -> boxes
# ============================================================

class Detections(NamedTuple):
    scores: torch.Tensor               # [batch, sites]
    boxes: torch.Tensor                # [batch, sites, 4]  xyxy in input pixels
    states: torch.Tensor               # [batch, sites]
    state_probabilities: torch.Tensor  # [batch, sites, states]


class GroundTruth(NamedTuple):
    present: torch.Tensor              # [batch, sites]  site is annotated
    boxes: torch.Tensor                # [batch, sites, 4]
    states: torch.Tensor               # [batch, sites]  -1 when not present


def _gather_at_cell(
    maps: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """maps [batch, sites, channels, grid, grid], index [batch, sites] -> [batch, sites, channels]."""
    flat = maps.flatten(3)
    index = index[:, :, None, None].expand(-1, -1, flat.size(2), 1)

    return flat.gather(3, index).squeeze(3)


def _cell_values_to_boxes(
    values: torch.Tensor,
    index: torch.Tensor,
    grid_size: int,
    image_size: int,
) -> torch.Tensor:
    """values [batch, sites, 4] = offset_x, offset_y, log_width, log_height -> xyxy boxes."""
    cell = image_size / grid_size
    rows = torch.div(index, grid_size, rounding_mode="floor").float()
    columns = (index % grid_size).float()

    center_x = (columns + values[..., 0]) * cell
    center_y = (rows + values[..., 1]) * cell
    width = values[..., 2].clamp(max=MAX_LOG_SIZE).exp() * cell
    height = values[..., 3].clamp(max=MAX_LOG_SIZE).exp() * cell

    return torch.stack(
        [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ],
        dim=-1,
    )


def decode_predictions(
    predictions: torch.Tensor,
    image_size: int,
) -> Detections:
    """Raw model output [batch, sites, 5 + states, grid, grid] -> one box per site."""
    grid_size = predictions.size(-1)

    heat = predictions[:, :, OBJECT].sigmoid()
    scores, index = heat.flatten(2).max(dim=2)

    values = _gather_at_cell(predictions[:, :, OFFSET_X:LOG_HEIGHT + 1], index)
    values = torch.cat([values[..., :2].sigmoid(), values[..., 2:]], dim=-1)

    state_probabilities = _gather_at_cell(predictions[:, :, STATE:], index).softmax(dim=-1)

    return Detections(
        scores=scores,
        boxes=_cell_values_to_boxes(values, index, grid_size, image_size),
        states=state_probabilities.argmax(dim=-1),
        state_probabilities=state_probabilities,
    )


def decode_targets(
    targets: torch.Tensor,
    image_size: int,
) -> GroundTruth:
    """Grid targets [batch, sites, 6, grid, grid] -> the original boxes and states."""
    grid_size = targets.size(-1)

    peak, index = targets[:, :, OBJECT].flatten(2).max(dim=2)

    values = _gather_at_cell(targets[:, :, OFFSET_X:LOG_HEIGHT + 1], index)
    states = _gather_at_cell(targets[:, :, STATE:STATE + 1], index).squeeze(-1)

    return GroundTruth(
        present=peak == 1.0,
        boxes=_cell_values_to_boxes(values, index, grid_size, image_size),
        states=states.long(),
    )