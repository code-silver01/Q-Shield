# Channel-Anomaly Model - Hardware/Raspberry Pi Handoff

This is the handoff package for the channel-anomaly Isolation Forest. It is
trained and working on synthetic data; it does **not** need to be retrained
by you. Your job is to produce telemetry in the format below so it can be
retrained on real data later and run live.

## What's in this package

| File | Purpose |
|---|---|
| `models/channel_iforest_v1.joblib` | The trained model. Load, don't edit. |
| `ml/feature_schema.json` | Exact feature names, order, units, ranges. Source of truth. |
| `ml/inference/detector.py` | `ChannelAnomalyDetector` - the only class you should call to get a verdict. |
| `ml/integration/pi_integration_example.py` | Reference code: raw packets -> 6 features -> verdict. |
| `ml/integration/sample_raw_telemetry.json` | Example of the raw packet log format described below. |
| `requirements.txt` | Pinned package versions (`pip install -r requirements.txt`). |

## The model does NOT take raw packets

It scores **one 5-second window** at a time, described by exactly 6 numbers.
You (or the gateway code) must aggregate raw packets into those 6 numbers
first - `ml/integration/pi_integration_example.py` shows exactly how.

### The 6 required features (per window, per channel)

| Feature | Unit | What it means |
|---|---|---|
| `packet_rate` | packets/second | packets received in the window / 5.0 |
| `inter_arrival_time` | seconds | mean gap between consecutive packet arrivals |
| `jitter` | seconds | standard deviation of those gaps |
| `packet_loss` | fraction 0.0-1.0 | (expected seq. numbers - received) / expected |
| `rejection_rate` | fraction 0.0-1.0 | crypto-rejected packets / received packets |
| `retransmissions` | count | number of packets flagged as resends |

Full definitions and assumed "normal" ranges are in `ml/feature_schema.json`
- treat that file as the contract, not this document.

**RSSI and SNR are intentionally NOT in v1.** Do not send them yet; we will
add a v2 schema once you have real values wired up.

## What you need to provide from the Pi / gateway

For **both** the primary channel and LoRa, log one record per received
packet with at least these fields:

```json
{
  "timestamp": 0.0512,      // seconds, relative to window start (or any monotonic clock)
  "seq_num": 17,            // sender's sequence number
  "status": "ok"            // "ok" | "retransmit" | "rejected_decrypt"
                              // | "rejected_signature" | "rejected_replay"
}
```

Group these into fixed windows (5 seconds is what the model assumes - see
`WINDOW_SECONDS` in `ml/features.py`) and feed each window's packet list into
`extract_features_from_window()` in `ml/integration/pi_integration_example.py`.
That function is the reference implementation - reuse it or port its exact
logic; if the gateway computes these features differently than training
assumed, the model's scores become meaningless.

If a window has **zero received packets** (e.g. total jamming), that's a
special case `extract_features_from_window()` raises on deliberately - decide
explicitly how your gateway should report that (e.g. `packet_rate=0`,
`packet_loss=1.0`, everything else `0`) rather than silently defaulting.

`packet_loss` is computed from **unique** sequence numbers received, not raw
packet count - a retransmitted packet resends an already-used sequence
number, so counting raw packets would let a retransmission hide an actual
loss elsewhere in the window. Keep this in mind if your Pi-side retransmit
logic reuses the original sequence number (the common case).

Windows are always per-channel: run primary-channel and LoRa telemetry
through this pipeline **separately** (tag each window's `channel` field
accordingly, as in the sample file) - never mix packets from both channels
into one window.

### Two data captures we'll need from you later

1. **Clean baseline traffic** (most important) - normal operation logs, at
   different distances/conditions, so the model can be retrained on real
   "normal" instead of synthetic normal.
2. **Labeled attack sessions** (jamming, flooding, etc.) with start/end
   timestamps - for evaluating the retrained model, not training it.

Log format: JSON or CSV, one record per packet, exact fields/units above.

## How to call the model from your integration code

```python
from ml.inference.detector import ChannelAnomalyDetector

detector = ChannelAnomalyDetector()  # loads models/channel_iforest_v1.joblib once

result = detector.score({
    "packet_rate": 9.8,
    "inter_arrival_time": 0.097,
    "jitter": 0.008,
    "packet_loss": 0.02,
    "rejection_rate": 0.02,
    "retransmissions": 2,
})
# {"anomaly_score": 0.092, "is_anomaly": False, "status": "NORMAL"}
```

`result["status"]` is `"NORMAL"` or `"ANOMALY"` - that's what should drive
Isolate -> Reroute-to-LoRa -> Recover on the gateway side.

## Verified working (this handoff)

Running `python ml/integration/pi_integration_example.py` against
`sample_raw_telemetry.json` (50 synthetic packets, primary channel, 2
retransmits, 1 crypto rejection, 2 sequence gaps):

```
Extracted features:
  packet_rate          = 10.0
  inter_arrival_time   = 0.0974
  jitter               = 0.0087
  packet_loss          = 0.0385
  rejection_rate       = 0.02
  retransmissions      = 2

Model result:
  status:        NORMAL
  is_anomaly:    False
  anomaly_score: 0.0636
```

Confirms: the model loads from `models/channel_iforest_v1.joblib`, the
feature extraction runs on raw packet-style input, and the detector returns
a correct NORMAL/ANOMALY verdict end-to-end.
