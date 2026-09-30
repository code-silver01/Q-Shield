"""
Inference wrapper for the Q-SHIELD channel-anomaly Isolation Forest.

Loads models/channel_iforest_v1.joblib and scores one telemetry window at a
time. Command/gateway code should call ONLY through this wrapper (not the
raw pipeline) so feature order and scaling always match training - see
ml/feature_schema.json for the exact input contract.

Usage:
    from ml.inference.detector import ChannelAnomalyDetector

    detector = ChannelAnomalyDetector()
    result = detector.score({
        "packet_rate": 10.2,
        "inter_arrival_time": 0.098,
        "jitter": 0.011,
        "packet_loss": 0.01,
        "rejection_rate": 0.0,
        "retransmissions": 0,
    })
    # result -> {"anomaly_score": float, "is_anomaly": bool, "status": "NORMAL"/"ANOMALY"}
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "ml"))
from features import FEATURE_ORDER

DEFAULT_MODEL_PATH = ROOT / "models" / "channel_iforest_v1.joblib"


class ChannelAnomalyDetector:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)

    def score(self, features: dict) -> dict:
        """features must contain exactly the keys in FEATURE_ORDER (any dict
        order is fine - this reindexes before scoring)."""
        missing = [f for f in FEATURE_ORDER if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        row = pd.DataFrame([[features[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

        raw_pred = self.pipeline.predict(row)[0]     # 1 = normal, -1 = anomaly
        score = float(self.pipeline.decision_function(row)[0])  # higher = more normal

        is_anomaly = bool(raw_pred == -1)
        return {
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "status": "ANOMALY" if is_anomaly else "NORMAL",
        }

    def score_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows must contain at least the FEATURE_ORDER columns."""
        rows = rows[FEATURE_ORDER]
        preds = self.pipeline.predict(rows)
        scores = self.pipeline.decision_function(rows)
        out = rows.copy()
        out["anomaly_score"] = scores
        out["is_anomaly"] = preds == -1
        out["status"] = np.where(out["is_anomaly"], "ANOMALY", "NORMAL")
        return out


if __name__ == "__main__":
    detector = ChannelAnomalyDetector()

    normal_sample = {
        "packet_rate": 10.2,
        "inter_arrival_time": 0.098,
        "jitter": 0.011,
        "packet_loss": 0.01,
        "rejection_rate": 0.0,
        "retransmissions": 0,
    }
    print("Sample NORMAL-ish input:", normal_sample)
    print("Result:", detector.score(normal_sample))

    flood_sample = {
        "packet_rate": 80.0,
        "inter_arrival_time": 0.0125,
        "jitter": 0.03,
        "packet_loss": 0.25,
        "rejection_rate": 0.02,
        "retransmissions": 4,
    }
    print("\nSample FLOODING-ish input:", flood_sample)
    print("Result:", detector.score(flood_sample))
