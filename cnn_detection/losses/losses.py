from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..grid import (
    LOG_HEIGHT,
    LOG_WIDTH,
    OBJECT,
    OFFSET_X,
    OFFSET_Y,
    STATE,
)


class DetectionLoss(nn.Module):
    """
    Four terms, each averaged over the positive (center) cells:

        object  focal loss on the objectness heatmap
        offset  L1 on the position inside the center cell
        size    L1 on log width / log height
        state   cross entropy on the state logits, class-weighted

    predictions: [batch, sites, 5 + states, grid, grid]  (raw)
    targets:     [batch, sites, 6, grid, grid]           (see grid.py)
    """

    def __init__(
        self,
        state_weights: torch.Tensor,
        coordinate_weight: float = 1.0,
        state_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.register_buffer("state_weights", state_weights)
        self.coordinate_weight = coordinate_weight
        self.state_weight = state_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_heat = targets[:, :, OBJECT]
        positive = target_heat == 1.0                          # [batch, sites, grid, grid]
        num_positive = positive.sum().clamp(min=1)

        # ---- objectness: focal loss, cells near a center are penalized less
        logits = predictions[:, :, OBJECT]
        probability = logits.sigmoid()

        positive_loss = -F.logsigmoid(logits) * (1.0 - probability) ** 2
        negative_loss = (
            -F.logsigmoid(-logits) * probability ** 2 * (1.0 - target_heat) ** 4
        )
        object_loss = torch.where(positive, positive_loss, negative_loss).sum() / num_positive

        # ---- box: only at the center cell
        mask = positive.unsqueeze(2)                           # [batch, sites, 1, grid, grid]

        offset_loss = (
            F.l1_loss(
                predictions[:, :, OFFSET_X:OFFSET_Y + 1].sigmoid(),
                targets[:, :, OFFSET_X:OFFSET_Y + 1],
                reduction="none",
            )
            * mask
        ).sum() / num_positive

        size_loss = (
            F.l1_loss(
                predictions[:, :, LOG_WIDTH:LOG_HEIGHT + 1],
                targets[:, :, LOG_WIDTH:LOG_HEIGHT + 1],
                reduction="none",
            )
            * mask
        ).sum() / num_positive

        # ---- state: normal / stenosis / ... at the center cell
        state_logits = predictions[:, :, STATE:].permute(0, 1, 3, 4, 2)[positive]   # [positives, states]
        state_targets = targets[:, :, STATE][positive].long()                       # [positives]

        if state_targets.numel() > 0:
            state_loss = F.cross_entropy(
                state_logits,
                state_targets,
                weight=self.state_weights,
            )
        else:
            state_loss = state_logits.sum() * 0.0

        total = (
            object_loss
            + self.coordinate_weight * (offset_loss + size_loss)
            + self.state_weight * state_loss
        )

        return {
            "total": total,
            "object": object_loss,
            "offset": offset_loss,
            "size": size_loss,
            "state": state_loss,
        }