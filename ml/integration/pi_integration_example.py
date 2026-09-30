"""
MINIMAL example: converts raw per-packet telemetry (the kind the Raspberry
Pi field node / gateway would log) into the six window-level features the
channel_iforest_v1 model expects, then scores that window with
ChannelAnomalyDetector.

This is a REFERENCE for the hardware teammate's actual logging/gateway
code - the feature math here is exactly what must be reproduced at runtime
(same formulas as ml/data/generate_synthetic_data.py's assumptions and
ml/feature_schema.json's definitions), or the model's output is meaningless.

Expected raw packet record (one per packet received in a window):
    {
        "timestamp": float,   # seconds since window start (or any monotonic clock)
        "seq_num": int,       # sender's packet sequence number
        "status": str         # one of: "ok", "retransmit",
                               #         "rejected_decrypt", "rejected_signature",
                               #         "rejected_replay"
    }

Usage:
    python ml/integration/pi_integration_example.py
    python ml/integration/pi_integration_example.py path/to/other_window.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.inference.detector import ChannelAnomalyDetector
from ml.features import WINDOW_SECONDS

DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent / "sample_raw_telemetry.json"

REJECTED_STATUSES = {"rejected_decrypt", "rejected_signature", "rejected_replay"}


def extract_features_from_window(packets: list[dict], window_seconds: float = WINDOW_SECONDS) -> dict:
    """Converts a list of raw per-packet records (one window, one channel)
    into the six features ChannelAnomalyDetector.score() expects.

    packets must be non-empty and sorted (or will be sorted here) by timestamp.
    """
    if not packets:
        raise ValueError("Cannot extract features from an empty packet window "
                          "(a fully-jammed window with zero received packets "
                          "must be handled explicitly upstream - see README).")

    packets = sorted(packets, key=lambda p: p["timestamp"])
    timestamps = np.array([p["timestamp"] for p in packets], dtype=float)
    seq_nums = np.array([p["seq_num"] for p in packets], dtype=int)
    statuses = [p["status"] for p in packets]

    n_received = len(packets)

    # packet_rate: packets received per second of the window.
    packet_rate = n_received / window_seconds

    # inter_arrival_time / jitter: mean / std of gaps between consecutive arrivals.
    if n_received >= 2:
        gaps = np.diff(timestamps)
        inter_arrival_time = float(np.mean(gaps))
        jitter = float(np.std(gaps))
    else:
        # Only one packet in the window - no gap to measure; fall back to the
        # window-implied spacing as a best-effort estimate.
        inter_arrival_time = window_seconds
        jitter = 0.0

    # packet_loss: fraction of expected sequence numbers never seen. Uses the
    # count of UNIQUE sequence numbers, not raw packet count - a retransmit
    # resends an already-used sequence number, so counting raw packets would
    # let a retransmission mask an actual loss elsewhere in the window.
    n_unique_seq = len(set(seq_nums.tolist()))
    expected = int(seq_nums.max() - seq_nums.min() + 1)
    packet_loss = max(0.0, (expected - n_unique_seq) / expected) if expected > 0 else 0.0

    # rejection_rate: fraction of received packets the crypto layer rejected.
    n_rejected = sum(1 for s in statuses if s in REJECTED_STATUSES)
    rejection_rate = n_rejected / n_received

    # retransmissions: count of packets flagged as resends.
    retransmissions = sum(1 for s in statuses if s == "retransmit")

    return {
        "packet_rate": packet_rate,
        "inter_arrival_time": inter_arrival_time,
        "jitter": jitter,
        "packet_loss": packet_loss,
        "rejection_rate": rejection_rate,
        "retransmissions": retransmissions,
    }


def main():
    sample_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE_PATH
    with open(sample_path) as f:
        window = json.load(f)

    print(f"Loaded raw telemetry window from: {sample_path}")
    print(f"Channel: {window['channel']}  |  window_seconds: {window['window_seconds']}")
    print(f"Raw packets in window: {len(window['packets'])}")

    features = extract_features_from_window(window["packets"], window["window_seconds"])
    print("\nExtracted features (input to the model):")
    for k, v in features.items():
        print(f"  {k:20s} = {v}")

    detector = ChannelAnomalyDetector()  # loads models/channel_iforest_v1.joblib
    result = detector.score(features)

    print("\nModel result:")
    print(f"  status:        {result['status']}")
    print(f"  is_anomaly:    {result['is_anomaly']}")
    print(f"  anomaly_score: {result['anomaly_score']}")


if __name__ == "__main__":
    main()
