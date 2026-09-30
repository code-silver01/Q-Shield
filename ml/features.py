"""Canonical feature contract for the channel-anomaly Isolation Forest.

Both training (training/train_model.py) and inference (inference/detector.py)
import FEATURE_ORDER from here so the model is never scored with columns in
a different order than it was trained on. See feature_schema.json for units
and descriptions.
"""

FEATURE_ORDER = [
    "packet_rate",
    "inter_arrival_time",
    "jitter",
    "packet_loss",
    "rejection_rate",
    "retransmissions",
]

# One sample = one summarized observation window on a single channel.
WINDOW_SECONDS = 5.0
