"""
Generates synthetic communication-channel telemetry for the Q-SHIELD
channel-anomaly Isolation Forest.

Each row represents one WINDOW_SECONDS-second observation window on one
channel (primary or LoRa), summarizing packets seen in that window - NOT a
single packet.

This is PLACEHOLDER data for initial model development only, until the
Raspberry Pi teammate provides real captured telemetry (see repo README).
All "typical" values below are assumptions, documented in feature_schema.json,
and must be replaced with statistics from real baseline captures later.

Run:
    python ml/data/generate_synthetic_data.py

Outputs (into this directory):
    normal_traffic.csv   - assumed-normal channel windows, used for TRAINING
    attack_traffic.csv   - labeled synthetic attack windows (column
                           'scenario'), used ONLY for evaluation, never
                           for training (Isolation Forest is unsupervised)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from features import FEATURE_ORDER

OUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42

N_NORMAL = 3000
N_PER_ATTACK_SCENARIO = 200

# Assumed baseline: ~10 packets/sec low-rate telemetry/heartbeat channel.
NORMAL_PACKET_RATE_MEAN = 10.0
NORMAL_PACKET_RATE_STD = 1.5


def _clip_nonneg(arr):
    return np.clip(arr, 0.0, None)


def _assemble(packet_rate, inter_arrival_time, jitter, packet_loss, rejection_rate, retransmissions):
    df = pd.DataFrame({
        "packet_rate": packet_rate,
        "inter_arrival_time": inter_arrival_time,
        "jitter": jitter,
        "packet_loss": packet_loss,
        "rejection_rate": rejection_rate,
        "retransmissions": retransmissions,
    })
    return df[FEATURE_ORDER]


def generate_normal(n, rng):
    packet_rate = np.clip(rng.normal(NORMAL_PACKET_RATE_MEAN, NORMAL_PACKET_RATE_STD, n), 1.0, None)
    inter_arrival_time = (1.0 / packet_rate) * rng.lognormal(0.0, 0.08, n)
    jitter = _clip_nonneg(rng.normal(0.01, 0.004, n))
    packet_loss = np.clip(rng.beta(1.0, 60.0, n), 0.0, 1.0)
    rejection_rate = np.clip(rng.beta(1.0, 100.0, n), 0.0, 1.0)
    retransmissions = rng.poisson(0.4, n).astype(float)
    return _assemble(packet_rate, inter_arrival_time, jitter, packet_loss, rejection_rate, retransmissions)


def generate_flooding(n, rng):
    # Attacker or misbehaving node blasts packets far above baseline rate.
    packet_rate = rng.uniform(40, 120, n)
    inter_arrival_time = (1.0 / packet_rate) * rng.lognormal(0.0, 0.15, n)
    jitter = _clip_nonneg(rng.normal(0.02, 0.01, n))
    packet_loss = np.clip(rng.beta(2.0, 8.0, n), 0, 1)      # congestion drops packets
    rejection_rate = np.clip(rng.beta(1.0, 40.0, n), 0, 1)
    retransmissions = rng.poisson(3.0, n).astype(float)
    return _assemble(packet_rate, inter_arrival_time, jitter, packet_loss, rejection_rate, retransmissions)


def generate_jamming(n, rng):
    # RF jamming / channel disruption: packets mostly fail to get through.
    packet_rate = np.clip(rng.normal(2.0, 1.5, n), 0.05, None)
    inter_arrival_time = np.clip(rng.normal(0.6, 0.3, n), 0.05, None)
    jitter = _clip_nonneg(rng.normal(0.25, 0.12, n))
    packet_loss = np.clip(rng.beta(6.0, 3.0, n), 0, 1)      # high loss
    rejection_rate = np.clip(rng.beta(1.0, 30.0, n), 0, 1)
    retransmissions = rng.poisson(6.0, n).astype(float)
    return _assemble(packet_rate, inter_arrival_time, jitter, packet_loss, rejection_rate, retransmissions)


def generate_loss_retransmit(n, rng):
    # Degraded link: near-normal rate/timing but heavy loss + retransmissions
    # and elevated crypto rejections (e.g. corrupted frames on a bad link).
    packet_rate = np.clip(rng.normal(9.0, 2.0, n), 1.0, None)
    inter_arrival_time = (1.0 / packet_rate) * rng.lognormal(0.0, 0.1, n)
    jitter = _clip_nonneg(rng.normal(0.015, 0.006, n))
    packet_loss = np.clip(rng.beta(5.0, 4.0, n), 0, 1)
    rejection_rate = np.clip(rng.beta(4.0, 6.0, n), 0, 1)
    retransmissions = rng.poisson(8.0, n).astype(float)
    return _assemble(packet_rate, inter_arrival_time, jitter, packet_loss, rejection_rate, retransmissions)


def generate_timing_jitter(n, rng):
    # Timing disruption: erratic inter-arrival/jitter, rate close to normal.
    packet_rate = np.clip(rng.normal(10.0, 3.0, n), 1.0, None)
    inter_arrival_time = np.clip(rng.normal(0.15, 0.12, n), 0.005, None)
    jitter = _clip_nonneg(rng.normal(0.12, 0.06, n))
    packet_loss = np.clip(rng.beta(1.5, 20.0, n), 0, 1)
    rejection_rate = np.clip(rng.beta(1.0, 60.0, n), 0, 1)
    retransmissions = rng.poisson(2.0, n).astype(float)
    return _assemble(packet_rate, inter_arrival_time, jitter, packet_loss, rejection_rate, retransmissions)


ATTACK_GENERATORS = {
    "flooding": generate_flooding,
    "jamming": generate_jamming,
    "loss_retransmit": generate_loss_retransmit,
    "timing_jitter": generate_timing_jitter,
}


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    normal_df = generate_normal(N_NORMAL, rng)
    normal_path = OUT_DIR / "normal_traffic.csv"
    normal_df.to_csv(normal_path, index=False)
    print(f"Wrote {len(normal_df)} normal windows -> {normal_path}")

    attack_frames = []
    for scenario, gen_fn in ATTACK_GENERATORS.items():
        df = gen_fn(N_PER_ATTACK_SCENARIO, rng)
        df["scenario"] = scenario
        attack_frames.append(df)
    attack_df = pd.concat(attack_frames, ignore_index=True)
    attack_path = OUT_DIR / "attack_traffic.csv"
    attack_df.to_csv(attack_path, index=False)
    print(f"Wrote {len(attack_df)} attack windows across {len(ATTACK_GENERATORS)} "
          f"scenarios -> {attack_path}")


if __name__ == "__main__":
    main()
