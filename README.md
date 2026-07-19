# EV Anomaly Detector

A two-stage anomaly-detection pipeline for electric-vehicle drive telemetry.
It ingests per-vehicle time-series telemetry (RPM, torque, speed, throttle,
current, SoC, controller temperature, …), flags vehicles whose behaviour
deviates from their peers, and explains *why* each flag fired.

The repo ships with a **synthetic data generator** so the whole pipeline runs
end-to-end out of the box — no database, no external data source.

## How it works

**Stage 1 — ML pattern detector.** Per vehicle-variant, builds a 7-signal
feature vector and runs KMeans (best-K by silhouette) + IsolationForest. A
vehicle is flagged if it's a cluster-distance outlier *or* an isolation-forest
outlier.

**Stage 2 — dual-signal contradictor.** Eight physics-based detectors look for
pairs of channels that contradict each other within a sustained dwell window —
e.g. *RPM high while speed stays low* (drivetrain slip), or *controller temp
high while current is low* (cooling fault). Vehicles are scored against
usage-matched peers within their variant.

**Verdict gate.** A vehicle is `ELEVATED` only when the ML signal **and** ≥2
strong physical detectors agree; weaker evidence is `WATCH`; clean vehicles with
valid trips are `NORMAL`. A spirited-driver guard prevents fast-but-healthy
vehicles from being flagged.

## Quick start

```bash
pip install -r requirements.txt

# 1. generate synthetic telemetry (writes data/demo_telemetry.csv)
python make_demo_data.py

# 2. run the full pipeline
python -m anomaly_detector.pipeline

# optional: also build + open the HTML dashboard preview
python -m anomaly_detector.pipeline --dashboard
```

Outputs land in `out/`:

| File | Contents |
|------|----------|
| `stage1_flags.json`   | Stage-1 ML flags per vehicle |
| `stage2_verdicts.json`| Final per-vehicle verdicts + evidence |
| `summary.json`        | Fleet-level counts and per-variant breakdown |

## Using your own data

Point the pipeline at any CSV with these columns and set `EV_INPUT_CSV`:

```
time, imei, odometer, rpm, torque, controllerTemperature, soc, speed,
throttle, current, vehicleState, controllerControllerMode,
vehicle_model_name, vehicle_variant_name
```

```bash
EV_INPUT_CSV=/path/to/your.csv python -m anomaly_detector.pipeline
```

All thresholds live in [`anomaly_detector/config.py`](anomaly_detector/config.py).

## Layout

```
anomaly_detector/
├── config.py            # all tunable thresholds
├── stage1_detector.py   # ML pattern detector (KMeans + IsolationForest)
├── stage2_validator.py  # dual-signal contradictor + verdict gate
├── pipeline.py          # end-to-end runner
├── explain.py           # per-vehicle evidence / explanations
├── fleet_reference.py   # peer reference table
├── peer_check.py        # interactive per-vehicle inspection
└── dashboard_*.py       # HTML preview
make_demo_data.py        # synthetic telemetry generator
```
