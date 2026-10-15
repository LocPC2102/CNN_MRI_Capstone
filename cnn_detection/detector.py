from __future__ import annotations

import torch
from torch import nn

from .backbone import LeNetBackbone


class LeNetDetector(nn.Module):
    """
    One detection slot per site (CCS, LLS, RLS, LFS, RFS).

    Every grid cell predicts, for every site:
        [objectness, offset_x, offset_y, log_width, log_height,
         state_0, ..., state_{num_states - 1}]

    Input:
        [batch, 1, 256, 256]

    Output:
        [batch, num_sites, 5 + num_states, 32, 32]
    """

    def __init__(
        self,
        num_sites: int,
        num_states: int,
    ) -> None:
        super().__init__()

        self.num_sites = num_sites
        self.num_states = num_states
        self.values_per_site = 5 + num_states

        self.backbone = LeNetBackbone()

        # 32x32x96 -> 32x32x(num_sites * (5 + num_states))
        #
        # This maps directly to a 1x1 detection head in RTL
        self.detection_head = nn.Conv2d(
            in_channels=self.backbone.out_channels,
            out_channels=num_sites * self.values_per_site,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self._initialize_objectness_bias()

    def _initialize_objectness_bias(self) -> None:
        # Start with "nothing here" (probability about 0.01) because
        # only one cell in 1024 holds an object per site.
        with torch.no_grad():
            bias = self.detection_head.bias.view(
                self.num_sites,
                self.values_per_site,
            )
            bias[:, 0] = -4.6

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(images)

        # Do not apply ReLU, sigmoid or softmax here.
        raw_predictions = self.detection_head(features)

        batch, _, height, width = raw_predictions.shape

        return raw_predictions.view(
            batch,
            self.num_sites,
            self.values_per_site,
            height,
            width,
        )