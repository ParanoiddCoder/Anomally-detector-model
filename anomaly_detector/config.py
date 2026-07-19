"""
config.py — central configuration for the EV anomaly pipeline.

Every tunable threshold lives here. Stage-1 and Stage-2 import from this file
so a single edit propagates through the whole pipeline. Anything not in here
is structural (algorithm shape) and should not be changed without a redesign.
"""
from __future__ import annotations

import os
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = Path(os.environ.get("EV_INPUT_CSV", ROOT / "data" / "demo_telemetry.csv"))
STAGE1_OUT = Path(os.environ.get("EV_STAGE1_OUT", ROOT / "out" / "stage1_flags.json"))
STAGE2_OUT = Path(os.environ.get("EV_STAGE2_OUT", ROOT / "out" / "stage2_verdicts.json"))
SUMMARY_OUT = Path(os.environ.get("EV_SUMMARY_OUT", ROOT / "out" / "summary.json"))


# ── Vehicle states (canonical) ─────────────────────────────────────────
STATE_IDLE = 0
STATE_CHARGE = 1
STATE_RUN = 2
STATE_LATCH = 3   # full-charge latch — NOT counted as charging time


# ── Model scope ────────────────────────────────────────────────────────
# Whitelist of vehicle_model_name values the pipeline will process.
# Telemetry rows for any other model are dropped at load time, so every
# downstream artefact (stage1 flags, stage2 verdicts, dashboard) is scoped
# to these models only. Set INCLUDE_MODELS = None to process everything.
INCLUDE_MODELS: tuple[str, ...] | None = None


# ── Telemetry / trip rules ─────────────────────────────────────────────
# Only moving rows (state==2 AND speed>0) are clustered into trips.
# A gap > TRIP_IDLE_GAP_SECONDS between consecutive moving rows starts a new
# trip cluster. Active duration counts only gaps ≤ TRIP_MAX_SAMPLE_GAP_SEC
# between moving rows — stops and data blackouts are excluded automatically.
# A cluster is kept only if active duration ≥ TRIP_MIN_DURATION_MIN and
# odometer delta ≥ TRIP_MIN_DISTANCE_KM.
TELEMETRY_SEC_PER_ROW = 5
TRIP_IDLE_GAP_SECONDS = 900        # gap between moving rows that starts a new trip
TRIP_MAX_SAMPLE_GAP_SEC = 120      # gaps larger than this are not counted as drive time
TRIP_GAP_SECONDS = TRIP_IDLE_GAP_SECONDS   # alias kept for charge-event segmentation
TRIP_MIN_DURATION_MIN = 3
TRIP_MIN_DISTANCE_KM = 2.0
# Sanity gate: km per 1% SOC consumed. Above this the odometer is rolling over
# or the SOC channel glitched (e.g. 2222 km in 22 min). Trip is rejected.
TRIP_MAX_KM_PER_SOC_PCT = 2.5


# ── Stage-1 ML detector ────────────────────────────────────────────────
# Per-variant KMeans + IsolationForest on 7-signal vehicle feature vectors.
# Stage-1 picks the best K per variant by silhouette score over [2, K_max].
STAGE1_KMEANS_K_MAX = 3
STAGE1_IFOREST_CONTAM = 0.10
STAGE1_RANDOM_SEED = 42
STAGE1_SIGNALS = ("rpm", "torque", "controllerTemperature", "soc",
                  "speed", "throttle", "current")
# A vehicle is is_anomaly if EITHER cluster-distance outlier OR iforest outlier.


# ── Stage-2 dual-signal contradictor (paired-channel detectors) ────────
DWELL_SAMPLES = 4                    # default dwell = 4 samples × 5s = 20s
DETECTOR_DWELL: dict[str, int] = {   # per-detector overrides
    "d1_rpm_speed": 3,               # drivetrain triad — high precision, accept 15s
    "d5_current_rpm_torque": 3,
    "d6_torque_speed": 3,
}

DUAL_DETECTORS: list[dict[str, str]] = [
    {"id": "d1_rpm_speed",          "label": "RPM high, Speed low (drivetrain slip)"},
    {"id": "d2_throttle_speed",     "label": "Throttle high, Speed low (demand without motion)"},
    {"id": "d3_throttle_current",   "label": "Throttle high, Current low (controller fault)"},
    {"id": "d4_current_torque",     "label": "Current high, Torque low (motor inefficiency)"},
    {"id": "d5_current_rpm_torque", "label": "Current+RPM high, Torque low (freewheel / driveline)"},
    {"id": "d6_torque_speed",       "label": "Torque high, Speed low (drag / brake-binding)"},
    {"id": "d7_charge_overtaper",   "label": "SoC high, Charge current high (BMS not tapering)"},
    {"id": "d8_temp_overheat",      "label": "Controller temp high, Current low (cooling fault)"},
]
DUAL_IDS: list[str] = [d["id"] for d in DUAL_DETECTORS]
METRIC_LABELS: dict[str, str] = {f"{d['id']}_rate": d["label"] for d in DUAL_DETECTORS}


# ── Peer matching + scoring ────────────────────────────────────────────
PEER_K = 20
PEER_MIN = 8
PHYS_Z_THRESH = 2.0          # peer-z threshold when ML flagged
UNFLAGGED_Z_THRESH = 2.5     # higher bar when ML did not flag

MATCH_FEATURES = (
    "mode_0_frac", "mode_1_frac",
    "mean_speed", "p90_speed",
    "median_torque", "p90_torque",
    "median_throttle", "p90_throttle",
    "low_speed_frac",
)


# ── Verdict gate (v5: ML-AND-physics required) ─────────────────────────
# ELEVATED requires BOTH ML signal AND ≥2 strong dual-signal detectors.
# Anything weaker is WATCH; pure absence of evidence (with valid trips) is NORMAL.
ELEVATED_DUAL_WITH_ML = 2     # tightened from 1 — ML alone is no longer enough
ELEVATED_DUAL_NO_ML = 99      # disabled — physics-only path no longer reaches ELEVATED


# ── Spirited-driver guard ──────────────────────────────────────────────
# Vehicles that simply drive fast (high speed + high rpm by intent) should not
# be flagged as anomalies. Demote to NORMAL when speed pattern is dominated by
# fast cruising AND physical evidence is weak.
# Carve-out: vehicles with n_dual ≥ 2 strong signals are NEVER demoted.
SPIRITED_TRIP_FRAC = 0.60          # ≥60% of trips have mean_speed > MEAN_SPEED_THR
SPIRITED_MEAN_SPEED_KMH = 40.0     # threshold per trip
SPIRITED_MEDIAN_P90_KMH = 55.0     # vehicle's median p90_speed across trips
SPIRITED_MAX_DUAL = 1              # only demote if n_dual ≤ this (preserves CONFIRMED)


# ── Fleet reference table ──────────────────────────────────────────────
FLEET_DURATION_BIN_MIN = 5         # bin width in minutes
FLEET_SOC_BIN_PCT = 2              # bin width in SOC % consumed (replaces dist_bin)
FLEET_MIN_BUCKET_N = 5             # below this, bucket is dropped
FLEET_LOW_CONFIDENCE_N = 10        # below this, bucket is flagged low_confidence


# ── Pipeline metadata ──────────────────────────────────────────────────
PIPELINE_VERSION = "v5-prod"
