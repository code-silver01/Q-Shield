"""
Trains the Q-SHIELD channel-anomaly Isolation Forest.

Usage:
    python ml/training/train_model.py

Reads:
    ml/data/normal_traffic.csv   (assumed-normal channel telemetry)
    ml/data/attack_traffic.csv   (labeled synthetic attacks, evaluation ONLY)

Writes:
    models/channel_iforest_v1.joblib   (StandardScaler + IsolationForest pipeline)

IMPORTANT: Isolation Forest is unsupervised anomaly detection. It is trained
ONLY on normal_traffic.csv. attack_traffic.csv's 'scenario' labels are used
exclusively to evaluate the trained model afterwards - they never influence
training.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "ml"))
from features import FEATURE_ORDER

DATA_DIR = ROOT / "ml" / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "channel_iforest_v1.joblib"

RANDOM_STATE = 42

# contamination: the fraction of TRAINING data IsolationForest expects to be
# anomalous. We train on data we generated as "normal", so this is set low
# (1%) mainly to absorb noisy/borderline synthetic samples rather than to
# model true real-world attack prevalence. Re-tune this against a real
# Raspberry Pi baseline capture once available.
CONTAMINATION = 0.01


def load_data():
    normal = pd.read_csv(DATA_DIR / "normal_traffic.csv")
    attacks = pd.read_csv(DATA_DIR / "attack_traffic.csv")
    return normal, attacks


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=200,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate(pipeline, normal_test, attacks):
    print("\n=== Evaluation: unseen NORMAL traffic (held-out split) ===")
    normal_pred = pipeline.predict(normal_test[FEATURE_ORDER])
    n_total = len(normal_pred)
    n_false_positive = int((normal_pred == -1).sum())
    print(f"Samples:                 {n_total}")
    print(f"Detected normal:         {n_total - n_false_positive}")
    print(f"Detected anomaly (FP):   {n_false_positive}")
    print(f"False positive rate:     {n_false_positive / n_total:.2%}")

    print("\n=== Evaluation: synthetic ATTACK traffic (by scenario) ===")
    for scenario, group in attacks.groupby("scenario"):
        pred = pipeline.predict(group[FEATURE_ORDER])
        n = len(pred)
        n_anomaly = int((pred == -1).sum())
        print(f"{scenario:20s} n={n:4d}  detected_anomaly={n_anomaly:4d}  "
              f"detection_rate={n_anomaly / n:.2%}")

    all_attack_pred = pipeline.predict(attacks[FEATURE_ORDER])
    overall_rate = (all_attack_pred == -1).mean()
    print(f"\nOverall attack detection rate (all scenarios combined): {overall_rate:.2%}")


def main():
    normal, attacks = load_data()

    normal_train, normal_test = train_test_split(
        normal, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(normal_train[FEATURE_ORDER])
    print(f"Trained IsolationForest on {len(normal_train)} normal windows "
          f"(contamination={CONTAMINATION}, random_state={RANDOM_STATE})")

    evaluate(pipeline, normal_test, attacks)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")

    print("\nNOTE: metrics above are on SYNTHETIC data only and do NOT "
          "represent real-world detection performance. Retrain/re-tune on "
          "real Raspberry Pi baseline + attack captures before drawing any "
          "conclusions about real deployment accuracy.")


if __name__ == "__main__":
    main()
