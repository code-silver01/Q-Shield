# Q-SHIELD - Channel Anomaly Detection (ML component)

Isolation Forest model that flags abnormal communication-channel behavior
(flooding, jamming/channel disruption, packet loss/retransmission storms,
timing/jitter disruption) from per-window telemetry summaries. Runs on the
laptop command/gateway side. Replay and tampering are handled by the
crypto layer (ML-KEM-768 / ML-DSA-65 nonces & signatures), not this model.

## Status

Trained on **synthetic data only** - no real Raspberry Pi telemetry yet.
Treat all metrics below as a pipeline smoke test, not a real-world accuracy
claim. Retrain on real baseline captures as soon as they exist (see
"What the hardware teammate needs to provide" below).

## How the synthetic dataset was generated

`ml/data/generate_synthetic_data.py` produces:

- `ml/data/normal_traffic.csv` - 3000 rows of assumed-normal channel windows
  (packet_rate ~ N(10, 1.5) pkt/s baseline with realistic noise on the other
  five features). Used for **training**.
- `ml/data/attack_traffic.csv` - 200 rows each for 4 attack scenarios
  (`flooding`, `jamming`, `loss_retransmit`, `timing_jitter`), labeled with a
  `scenario` column. Used **only for evaluation**, never for training.

Each row = one 5-second observation window on one channel, not a single
packet. Exact feature definitions, units, and assumed ranges are documented
in `ml/feature_schema.json`.

Re-run generation any time with:

```
python ml/data/generate_synthetic_data.py
```

## How the model was trained

Isolation Forest is unsupervised, so it is trained **only** on
`normal_traffic.csv`; the labeled attack file is used purely to evaluate the
trained model afterward. Preprocessing (`StandardScaler`) and the model
(`IsolationForest`) are bundled into a single `sklearn.Pipeline` so they can
never get out of sync.

- `n_estimators=200`, `contamination=0.01`, `random_state=42`
- `contamination=0.01` assumes ~1% of the synthetic "normal" windows are
  noisy/borderline - it is not an estimate of real attack prevalence. Re-tune
  it against real baseline data.

### Run training

```
python ml/training/train_model.py
```

This loads the two CSVs above, does an 80/20 split of `normal_traffic.csv`
(train / held-out normal test), fits the pipeline, prints evaluation
metrics, and saves the model.

### Evaluation results (synthetic data, this run)

| Set                    | n   | Detected anomaly | Rate    |
|-------------------------|-----|-------------------|---------|
| Held-out normal (FP)    | 600 | 6                 | 1.00%   |
| flooding                | 200 | 197               | 98.50%  |
| jamming                 | 200 | 200               | 100.00% |
| loss_retransmit         | 200 | 200               | 100.00% |
| timing_jitter           | 200 | 179               | 89.50%  |
| **Overall attack detection** | 800 | 776         | **97.00%** |

These numbers reflect how well the model separates the *synthetic*
distributions from each other - they say nothing about real-world
performance until retrained on real hardware data.

## Where the trained model is

`models/channel_iforest_v1.joblib` - a single `sklearn.Pipeline`
(`StandardScaler` + `IsolationForest`). Small (~KB-MB), committed directly
to the repo - no Git LFS needed.

## How the inference function is used

```python
from ml.inference.detector import ChannelAnomalyDetector

detector = ChannelAnomalyDetector()  # loads models/channel_iforest_v1.joblib
result = detector.score({
    "packet_rate": 10.2,
    "inter_arrival_time": 0.098,
    "jitter": 0.011,
    "packet_loss": 0.01,
    "rejection_rate": 0.0,
    "retransmissions": 0,
})
# {"anomaly_score": 0.214, "is_anomaly": False, "status": "NORMAL"}
```

`score_batch(df)` scores a whole DataFrame of windows at once. Feature
names/order/units are the single source of truth in `ml/feature_schema.json`
and `ml/features.py` - the gateway must compute these six features the same
way at runtime as they're generated here, or scores will be meaningless.

Run the file directly for a quick sanity check:

```
python ml/inference/detector.py
```

## What the Raspberry Pi / hardware teammate must eventually provide

Per-window (or per-packet, aggregated into windows) telemetry for **both**
the primary channel and LoRa, covering:

- Timestamps / packet arrival times (to compute `inter_arrival_time`, `jitter`)
- Sequence numbers (to compute `packet_loss`)
- Retransmission/ACK counts (`retransmissions`)
- Crypto-layer failure counts: failed decryptions, failed ML-DSA signatures,
  rejected/replayed nonces (`rejection_rate`)
- Packet counts per window (`packet_rate`)
- Later (v2 schema): RSSI (both channels) and SNR (LoRa) - intentionally
  excluded from v1 since no real values exist yet

Two kinds of captures are needed:

1. **Clean baseline traffic** (most important) - normal operation across
   different distances/conditions, used to retrain the "normal" model on
   real data.
2. **Labeled attack sessions** (jamming, flooding, etc.) with start/end
   times - used to re-evaluate, not to train.

Once real data arrives: regenerate `normal_traffic.csv` from real baseline
captures (keep the same 6 columns/units from `feature_schema.json`), rerun
`train_model.py`, and re-tune `contamination` / the anomaly threshold before
treating this as production-ready.
