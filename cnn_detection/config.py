from __future__ import annotations

from pathlib import Path

# ============================================================
# PATHS
# ============================================================

SPLIT_ROOT = Path("DATA/processed/detection")
CLASS_MAP_PATH = Path("DATA/metadata/detection_class_map.json")

MODEL_DIRECTORY = Path("src/cnn_detection/checkpoints")
MODEL_PATH = MODEL_DIRECTORY / "lenet_detector_best.pt"

# ============================================================
# INPUT & GRID
# ============================================================

IMAGE_SIZE = 256
# ============================================================
# AUGMENTATION (chỉ dùng khi train)
# ============================================================

MAX_SHIFT = 10                  # pixel, dịch ảnh tối đa
SCALE_RANGE = (0.9, 1.1)        # zoom nhẹ
GAMMA_RANGE = (0.8, 1.25)       # đổi độ sáng nhẹ
FLIP_PROBABILITY = 0.5          # đặt 0 để tắt lật ngang

# Sau khi lật ngang, trái thành phải
SITES = ("CCS", "LLS", "RLS", "LFS", "RFS")
STATES = ("normal", "stenosis", "severe_stenosis")
MIRROR_PERMUTATION = [0, 2, 1, 4, 3]


# Must match LeNetBackbone: three 2x2 poolings -> stride 8
STRIDE = 8
GRID_SIZE = IMAGE_SIZE // STRIDE

# ============================================================
# LABELS
#
# A class name in the XML is "<SITE>_<STATE>", e.g. LFS_stenosis.
# Each image has at most one object per site, so the model keeps
# one slot per site: it finds the box, then classifies the state.
# Check STATES against DATA/metadata/detection_class_map.json
# ============================================================

SITES = ("CCS", "LLS", "RLS", "LFS", "RFS")
STATES = ("normal", "stenosis", "severe_stenosis")

NUM_SITES = len(SITES)
NUM_STATES = len(STATES)

# objectness, offset_x, offset_y, log_width, log_height, then one logit per state
VALUES_PER_SITE = 5 + NUM_STATES

# ============================================================
# TRAINING
# ============================================================

RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

COORDINATE_WEIGHT = 1.0
STATE_WEIGHT = 1.0

# ============================================================
# EVALUATION
# ============================================================

SCORE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5