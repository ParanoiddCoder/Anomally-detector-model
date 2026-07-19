"""
stage2_validator.py — Stage-2 Dual-Signal Contradictor (production).

Consumes Stage-1 output (`stage1_flags.json`) + raw telemetry, emits a single
verdict per vehicle: ELEVATED / WATCH / NORMAL.

What this script does (and does NOT do):
  Keeps:
    1. Per-variant percentile reference stats (peer-relative thresholds).
    2. 8 dual-relationship detectors (d1..d8) with per-detector dwell.
    3. Peer matching (k=20 usage-matched peers within variant).
    4. Peer-z scoring with severity-by-rate OR gate.
    5. Trip segmentation (vehicleState==2, speed>0, gap>300s, ≥5min, ≥3km).
    6. Charging-session scan for d7.
    7. Single-tier verdict gate: ELEVATED = (ML+≥1 dual) OR ≥3 dual.

  Removed (no measurable lift on demo cohort, kept code complexity high):
    - Timeline / CUSUM change-point detection per signal.
    - Trip stability ratio.
    - Multi-model consensus (IForest-vehicle, IForest-trip, KMeans regimes).
    - Per-trip detector co-firing audit.
    - Six-section explanation panels — replaced with one compact reason string.
    - Excel writer — output is JSON only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as C


# ── Data loaders ───────────────────────────────────────────────────────
def _normalize_imei(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    try:
        n = float(text)
        if math.isfinite(n) and n.is_integer():
            return str(int(n))
    except ValueError:
        pass
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def load_stage1(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    df = pd.DataFrame(payload.get("vehicles", []))
    if df.empty:
        return df
    df["imei"] = df["imei"].map(_normalize_imei).astype("string")
    df["detector_flag"] = df["is_anomaly"].astype(bool)
    return df[["imei", "variant", "detector_flag", "isolation_score",
               "distance_to_center", "cluster_id", "severity"]]


def load_telemetry(path: Path) -> pd.DataFrame:
    usecols = ["time", "odometer", "imei", "rpm", "torque",
               "controllerTemperature", "soc", "speed", "throttle", "current",
               "vehicleState", "controllerControllerMode",
               "vehicle_model_name", "vehicle_variant_name"]
    # Per-chunk type conversion + downcast keeps peak memory bounded to one
    # chunk's worth instead of the full raw CSV materialised as object dtype.
    float_cols = ("odometer", "rpm", "torque", "controllerTemperature", "soc",
                  "speed", "throttle", "current")
    int_cols = ("vehicleState", "controllerControllerMode")
    soc_needs_scale = False
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000, low_memory=False):
        chunk = chunk.dropna(subset=["vehicle_model_name", "vehicle_variant_name"])
        if C.INCLUDE_MODELS is not None:
            chunk = chunk[chunk["vehicle_model_name"].isin(C.INCLUDE_MODELS)]
        if chunk.empty:
            continue
        chunk["imei"] = chunk["imei"].map(_normalize_imei).astype("string")
        chunk = chunk.dropna(subset=["imei"])
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["time"])
        if chunk.empty:
            continue
        for col in float_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")
        for col in int_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int16")
        chunk["variant"] = (chunk["vehicle_model_name"].astype(str)
                            + " | " + chunk["vehicle_variant_name"].astype(str))
        chunk = chunk.drop(columns=["vehicle_model_name", "vehicle_variant_name"])
        s = chunk["soc"].dropna()
        if len(s) and s.max() <= 1.5:
            soc_needs_scale = True
        parts.append(chunk)
    if not parts:
        scope = list(C.INCLUDE_MODELS) if C.INCLUDE_MODELS else "any model"
        raise ValueError(f"No rows for {scope} in telemetry CSV.")
    df = pd.concat(parts, ignore_index=True)
    if soc_needs_scale:
        df["soc"] = df["soc"].astype("float32") * 100
    return df


# ── Per-variant reference stats ────────────────────────────────────────
def compute_variant_stats(telemetry: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-variant percentiles used as peer-relative thresholds by all detectors."""
    stats: dict[str, dict[str, float]] = {}
    for variant, vdf in telemetry.groupby("variant", sort=False):
        drive = vdf[vdf["vehicleState"] == C.STATE_RUN]
        charge = vdf[vdf["vehicleState"] == C.STATE_CHARGE]
        src = drive if len(drive) >= 500 else vdf
        entry: dict[str, float] = {}

        for col in ("rpm", "torque", "current", "throttle"):
            s = pd.to_numeric(src[col], errors="coerce").dropna()
            if not len(s):
                continue
            entry[f"{col}_p25"] = float(s.quantile(0.25))
            entry[f"{col}_p50"] = float(s.quantile(0.50))
            entry[f"{col}_p75"] = float(s.quantile(0.75))
            entry[f"{col}_p90"] = float(s.quantile(0.90))

        sp = pd.to_numeric(src["speed"], errors="coerce").dropna()
        if len(sp):
            entry["speed_p25"] = float(sp.quantile(0.25))

        if "controllerTemperature" in src.columns:
            tt = pd.to_numeric(src["controllerTemperature"], errors="coerce").dropna()
            if len(tt):
                entry["temp_p90"] = float(tt.quantile(0.90))

        rs = src[(src["speed"] > 3) & (src["rpm"] > 50)]
        if len(rs):
            ratio = (pd.to_numeric(rs["rpm"], errors="coerce")
                     / pd.to_numeric(rs["speed"], errors="coerce")).dropna()
            if len(ratio):
                entry["ratio_rpm_speed_p95"] = float(ratio.quantile(0.95))

        if len(charge) >= 200:
            cc = pd.to_numeric(charge["current"], errors="coerce").dropna()
            if len(cc):
                entry["charge_current_p75"] = float(cc.quantile(0.75))

        stats[variant] = entry
    return stats


# ── Dual-relationship detectors ────────────────────────────────────────
def _events_from_mask(mask: np.ndarray, dwell: int) -> dict[str, float]:
    if len(mask) == 0:
        return {"events": 0, "max_dwell_s": 0.0, "mean_dwell_s": 0.0}
    m = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], m, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    runs = ends - starts
    qual = runs[runs >= dwell]
    if len(qual) == 0:
        return {"events": 0, "max_dwell_s": 0.0, "mean_dwell_s": 0.0}
    secs = qual * float(C.TELEMETRY_SEC_PER_ROW)
    return {"events": int(len(qual)), "max_dwell_s": float(secs.max()), "mean_dwell_s": float(secs.mean())}


def _dwell(det_id: str) -> int:
    return C.DETECTOR_DWELL.get(det_id, C.DWELL_SAMPLES)


def compute_dual_events(trip: pd.DataFrame, vstats: dict[str, float], state: int) -> dict[str, dict[str, float]]:
    out = {d: {"events": 0, "max_dwell_s": 0.0, "mean_dwell_s": 0.0} for d in C.DUAL_IDS}
    if trip.empty or not vstats:
        return out

    rpm = pd.to_numeric(trip["rpm"], errors="coerce").to_numpy()
    speed = pd.to_numeric(trip["speed"], errors="coerce").to_numpy()
    torque = pd.to_numeric(trip["torque"], errors="coerce").to_numpy()
    current = pd.to_numeric(trip["current"], errors="coerce").to_numpy()
    throttle = pd.to_numeric(trip["throttle"], errors="coerce").to_numpy()
    soc = pd.to_numeric(trip["soc"], errors="coerce").to_numpy() if "soc" in trip.columns else np.full(len(trip), np.nan)
    temp = (pd.to_numeric(trip["controllerTemperature"], errors="coerce").to_numpy()
            if "controllerTemperature" in trip.columns else np.full(len(trip), np.nan))

    finite = np.isfinite

    if state == C.STATE_RUN:
        ratio_thr = vstats.get("ratio_rpm_speed_p95")
        if ratio_thr is not None:
            safe = np.where(speed > 3, speed, np.nan)
            r = rpm / safe
            mask = finite(r) & (speed > 3) & (rpm > 50) & (r > ratio_thr)
            out["d1_rpm_speed"] = _events_from_mask(mask, _dwell("d1_rpm_speed"))

        thr_p75 = vstats.get("throttle_p75")
        sp_p25 = vstats.get("speed_p25")
        if thr_p75 is not None and sp_p25 is not None:
            mask = finite(throttle) & finite(speed) & (throttle >= thr_p75) & (speed <= sp_p25)
            out["d2_throttle_speed"] = _events_from_mask(mask, _dwell("d2_throttle_speed"))

        cur_p25 = vstats.get("current_p25")
        if thr_p75 is not None and cur_p25 is not None:
            mask = finite(throttle) & finite(current) & (throttle >= thr_p75) & (current <= cur_p25)
            out["d3_throttle_current"] = _events_from_mask(mask, _dwell("d3_throttle_current"))

        cur_p75 = vstats.get("current_p75")
        trq_p25 = vstats.get("torque_p25")
        if cur_p75 is not None and trq_p25 is not None:
            mask = finite(current) & finite(torque) & (current >= cur_p75) & (torque <= trq_p25)
            out["d4_current_torque"] = _events_from_mask(mask, _dwell("d4_current_torque"))

        rpm_p75 = vstats.get("rpm_p75")
        trq_p50 = vstats.get("torque_p50")
        if cur_p75 is not None and rpm_p75 is not None and trq_p50 is not None:
            mask = (finite(current) & finite(rpm) & finite(torque)
                    & (current >= cur_p75) & (rpm >= rpm_p75) & (torque <= trq_p50))
            out["d5_current_rpm_torque"] = _events_from_mask(mask, _dwell("d5_current_rpm_torque"))

        trq_p90 = vstats.get("torque_p90")
        if trq_p90 is not None and sp_p25 is not None:
            mask = finite(torque) & finite(speed) & (torque >= trq_p90) & (speed <= sp_p25)
            out["d6_torque_speed"] = _events_from_mask(mask, _dwell("d6_torque_speed"))

        temp_p90 = vstats.get("temp_p90")
        if temp_p90 is not None and cur_p25 is not None:
            mask = finite(temp) & finite(current) & (temp >= temp_p90) & (current <= cur_p25)
            out["d8_temp_overheat"] = _events_from_mask(mask, _dwell("d8_temp_overheat"))

    elif state == C.STATE_CHARGE:
        ch_p75 = vstats.get("charge_current_p75")
        if ch_p75 is not None:
            mask = finite(soc) & finite(current) & (soc >= 95.0) & (current >= ch_p75)
            out["d7_charge_overtaper"] = _events_from_mask(mask, _dwell("d7_charge_overtaper"))

    return out


# ── Trip table ─────────────────────────────────────────────────────────
def _segment_trips(vdf: pd.DataFrame) -> pd.DataFrame:
    """Return only moving discharge rows, labelled with a trip_id.

    Clustering logic (mirrors reference pipeline):
      1. Keep only state==2 AND speed>0 rows (pure discharge, actually moving).
      2. A gap > TRIP_IDLE_GAP_SECONDS between consecutive moving rows starts
         a new trip cluster — stops and overnight breaks split trips naturally.
    """
    sorted_v = vdf.sort_values("time").reset_index(drop=True)
    moving = sorted_v[
        (sorted_v["vehicleState"] == C.STATE_RUN) & (sorted_v["speed"].fillna(0) > 0)
    ].copy()
    if moving.empty:
        moving["trip_id"] = pd.Series(dtype="Int64")
        return moving
    gap = moving["time"].diff().dt.total_seconds().gt(C.TRIP_IDLE_GAP_SECONDS).fillna(True)
    moving["trip_id"] = gap.cumsum().astype(int)
    return moving


def _active_duration_min(trip: pd.DataFrame) -> float:
    """Sum of inter-row gaps where the earlier row is moving and gap ≤ MAX_SAMPLE_GAP_SEC.

    This excludes stops (no moving rows ⟹ gap not counted) and data blackouts
    (large gap capped out), so duration reflects real drive time only.
    All rows in `trip` are already speed>0 (from _segment_trips).
    """
    times = trip["time"].sort_values().reset_index(drop=True)
    if len(times) < 2:
        return 0.0
    gaps = times.diff().dt.total_seconds().iloc[1:]   # forward gaps, skip NaN at index 0
    active_sec = float(gaps[gaps <= C.TRIP_MAX_SAMPLE_GAP_SEC].sum())
    return round(active_sec / 60.0, 4)


def _trip_metrics(trip: pd.DataFrame, vstats: dict[str, float]) -> dict[str, Any] | None:
    if len(trip) < 2:
        return None

    duration_min = _active_duration_min(trip)
    if duration_min < C.TRIP_MIN_DURATION_MIN:
        return None

    odo = pd.to_numeric(trip["odometer"], errors="coerce").dropna()
    if len(odo) < 2:
        return None
    dist_km = float(odo.iloc[-1] - odo.iloc[0])
    if dist_km < C.TRIP_MIN_DISTANCE_KM:
        return None

    speed = pd.to_numeric(trip["speed"], errors="coerce")
    rpm = pd.to_numeric(trip["rpm"], errors="coerce")
    torque = pd.to_numeric(trip["torque"], errors="coerce")
    throttle = pd.to_numeric(trip["throttle"], errors="coerce")
    current = pd.to_numeric(trip["current"], errors="coerce")
    soc = pd.to_numeric(trip["soc"], errors="coerce") if "soc" in trip.columns else pd.Series(dtype=float)
    temp = (pd.to_numeric(trip["controllerTemperature"], errors="coerce")
            if "controllerTemperature" in trip.columns else pd.Series(dtype=float))
    mode = pd.to_numeric(trip["controllerControllerMode"], errors="coerce")

    soc_start = float(soc.dropna().iloc[0]) if len(soc.dropna()) else float("nan")
    soc_end = float(soc.dropna().iloc[-1]) if len(soc.dropna()) else float("nan")
    soc_consumed = (soc_start - soc_end) if (np.isfinite(soc_start) and np.isfinite(soc_end)) else float("nan")
    km_per_soc_pct = (dist_km / soc_consumed) if (np.isfinite(soc_consumed) and soc_consumed > 0) else float("nan")
    if np.isfinite(km_per_soc_pct) and km_per_soc_pct > C.TRIP_MAX_KM_PER_SOC_PCT:
        return None
    peak_temp = float(temp.max(skipna=True)) if len(temp) else float("nan")
    temp_delta_c = float(temp.max(skipna=True) - temp.min(skipna=True)) if len(temp.dropna()) >= 2 else float("nan")
    median_current = float(current.median(skipna=True)) if len(current) else float("nan")
    avg_speed = float(speed.mean(skipna=True))
    current_per_speed = (median_current / avg_speed) if (np.isfinite(median_current) and avg_speed > 0) else float("nan")
    median_torque = float(torque.median(skipna=True))
    current_per_torque = (median_current / median_torque) if (np.isfinite(median_current) and median_torque > 0) else float("nan")
    n_rows = max(len(trip), 1)
    current_spike_rate = float((current > 150).sum()) / n_rows if len(current) else float("nan")
    high_temp_rate = float((temp > 50).sum()) / n_rows if len(temp) else float("nan")

    metrics: dict[str, Any] = {
        "duration_min": float(duration_min),
        "dist_km": float(dist_km),
        "mean_speed": avg_speed,
        "avg_speed": avg_speed,                              # alias used by dashboard
        "p90_speed": float(speed.quantile(0.90)),
        "median_rpm": float(rpm.median(skipna=True)) if len(rpm.dropna()) else float("nan"),
        "p90_rpm": float(rpm.quantile(0.90)) if len(rpm.dropna()) else float("nan"),
        "median_torque": median_torque,
        "p90_torque": float(torque.quantile(0.90)),
        "median_throttle": float(throttle.median(skipna=True)),
        "p90_throttle": float(throttle.quantile(0.90)),
        "low_speed_frac": float((speed < 10).mean()),
        "median_current": median_current,
        "p90_current": float(current.quantile(0.90)) if len(current.dropna()) else float("nan"),
        "median_temp": float(temp.median(skipna=True)) if len(temp.dropna()) else float("nan"),
        "soc_start": soc_start,
        "soc_end": soc_end,
        "soc_consumed": soc_consumed,
        "km_per_soc_pct": km_per_soc_pct,
        "peak_temp": peak_temp,
        "temp_delta_c": temp_delta_c,
        "current_per_speed": current_per_speed,
        "current_per_torque": current_per_torque,
        "current_spike_rate": current_spike_rate,
        "high_temp_rate": high_temp_rate,
    }
    # Only mode 0 (Eco) and 1 (Thunder) are real — anything else is garbage
    # from a broken sensor read. Drop those rows from the denominator so the
    # fractions reflect real driving time and sum to 1.0.
    valid_mode = mode[mode.isin([0, 1])]
    n_valid = len(valid_mode)
    metrics["mode_0_frac"] = float((valid_mode == 0).mean()) if n_valid else 0.0
    metrics["mode_1_frac"] = float((valid_mode == 1).mean()) if n_valid else 0.0

    dual = compute_dual_events(trip, vstats, C.STATE_RUN)
    for det_id, ev in dual.items():
        metrics[f"{det_id}_events"] = int(ev["events"])
        metrics[f"{det_id}_max_dwell_s"] = float(ev["max_dwell_s"])
    return metrics


def build_trip_table(telemetry: pd.DataFrame, variant_stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, imei), vdf in telemetry.groupby(["variant", "imei"], sort=False):
        active = _segment_trips(vdf)
        if active.empty:
            continue
        vstats = variant_stats.get(variant, {})
        for trip_id, trip in active.groupby("trip_id", sort=True):
            m = _trip_metrics(trip, vstats)
            if m is None:
                continue
            m.update({"variant": variant, "imei": imei, "trip_id": int(trip_id)})
            rows.append(m)
    return pd.DataFrame(rows)


def build_charge_events(telemetry: pd.DataFrame, variant_stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, imei), vdf in telemetry.groupby(["variant", "imei"], sort=False):
        vstats = variant_stats.get(variant, {})
        charge = vdf[vdf["vehicleState"] == C.STATE_CHARGE].sort_values("time").reset_index(drop=True)
        if charge.empty:
            rows.append({"variant": variant, "imei": imei,
                         "d7_events": 0, "d7_max_dwell_s": 0.0, "charge_hours": 0.0})
            continue
        gaps = charge["time"].diff().dt.total_seconds()
        session_id = (gaps.isna() | (gaps > C.TRIP_GAP_SECONDS)).cumsum()
        charge["session_id"] = session_id

        total_events = 0
        max_dwell = 0.0
        total_hours = 0.0
        for _, sess in charge.groupby("session_id", sort=True):
            if len(sess) < C.DWELL_SAMPLES:
                continue
            secs = (sess["time"].iloc[-1] - sess["time"].iloc[0]).total_seconds()
            total_hours += max(secs, 0.0) / 3600.0
            ev = compute_dual_events(sess, vstats, C.STATE_CHARGE).get(
                "d7_charge_overtaper", {"events": 0, "max_dwell_s": 0.0})
            total_events += int(ev["events"])
            max_dwell = max(max_dwell, float(ev["max_dwell_s"]))
        rows.append({"variant": variant, "imei": imei,
                     "d7_events": int(total_events),
                     "d7_max_dwell_s": float(max_dwell),
                     "charge_hours": float(total_hours)})
    return pd.DataFrame(rows)


# ── Vehicle profiles, peers, scoring ───────────────────────────────────
def build_profiles(trips: pd.DataFrame, charges: pd.DataFrame) -> pd.DataFrame:
    charge_lookup = {(r["variant"], r["imei"]): r for _, r in charges.iterrows()}
    rows: list[dict[str, Any]] = []
    for (variant, imei), g in trips.groupby(["variant", "imei"], sort=False):
        row: dict[str, Any] = {"variant": variant, "imei": imei, "n_trips": int(len(g))}
        for col in C.MATCH_FEATURES:
            row[col] = float(g[col].median()) if col in g else np.nan
        drive_hours = float(g["duration_min"].sum() / 60.0) if "duration_min" in g else 0.0
        row["drive_hours"] = drive_hours

        for det_id in C.DUAL_IDS:
            if det_id == "d7_charge_overtaper":
                continue
            ev_col = f"{det_id}_events"
            mx_col = f"{det_id}_max_dwell_s"
            total = float(g[ev_col].sum()) if ev_col in g else 0.0
            row[f"{det_id}_rate"] = (total / drive_hours) if drive_hours > 0 else 0.0
            row[f"{det_id}_events_total"] = int(total)
            row[f"{det_id}_max_dwell_s"] = float(g[mx_col].max()) if mx_col in g else 0.0

        ch = charge_lookup.get((variant, imei))
        ch_hours = float(ch["charge_hours"]) if ch is not None else 0.0
        ch_events = int(ch["d7_events"]) if ch is not None else 0
        row["d7_charge_overtaper_rate"] = (ch_events / ch_hours) if ch_hours > 0 else 0.0
        row["d7_charge_overtaper_events_total"] = ch_events
        row["d7_charge_overtaper_max_dwell_s"] = float(ch["d7_max_dwell_s"]) if ch is not None else 0.0
        row["charge_hours"] = ch_hours
        rows.append(row)
    return pd.DataFrame(rows)


def _robust_scale(s: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    med = float(s.median())
    mad = float((s - med).abs().median()) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        iqr = float(s.quantile(0.75) - s.quantile(0.25))
        mad = iqr / 1.349 if iqr > 0 else float(s.std(ddof=0) or 1.0)
    return med, max(mad, 1e-9)


def find_peers(target: pd.Series, variant_pool: pd.DataFrame) -> pd.DataFrame:
    cands = variant_pool[variant_pool["imei"] != target["imei"]].copy()
    if len(cands) < C.PEER_MIN:
        return cands.iloc[0:0]
    dist = np.zeros(len(cands), dtype=float)
    for f in C.MATCH_FEATURES:
        all_vals = pd.concat([cands[f], pd.Series([target[f]])], ignore_index=True)
        med, scale = _robust_scale(all_vals)
        tv = target[f]
        tv = med if not np.isfinite(tv) else tv
        cv = cands[f].fillna(med)
        dist += ((cv - tv) / scale).to_numpy(dtype=float) ** 2
    cands["_d"] = np.sqrt(dist)
    return cands.sort_values("_d").head(min(C.PEER_K, len(cands)))


def score_vehicle(target: pd.Series, peers: pd.DataFrame, threshold: float) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Returns (residuals, strong_metric_ids). Strong = peer-z>=thr OR rate>=peer_p95."""
    residuals: dict[str, dict[str, Any]] = {}
    strong: list[str] = []
    for det_id in C.DUAL_IDS:
        metric = f"{det_id}_rate"
        value = target.get(metric, np.nan)
        peer_rates = pd.to_numeric(peers[metric], errors="coerce").dropna()

        z: float | None = None
        if np.isfinite(value) and len(peer_rates) >= C.PEER_MIN:
            med, scale = _robust_scale(peer_rates)
            z = float((float(value) - med) / scale)

        z_strong = z is not None and z >= threshold
        peer_p95 = float(peer_rates.quantile(0.95)) if len(peer_rates) else float("nan")
        rate_strong = (np.isfinite(value) and float(value) > 0
                       and np.isfinite(peer_p95) and float(value) >= peer_p95)

        is_strong = bool(z_strong or rate_strong)
        if is_strong:
            strong.append(metric)
        residuals[metric] = {
            "z": z,
            "value": float(value) if np.isfinite(value) else None,
            "peer_p95": peer_p95 if np.isfinite(peer_p95) else None,
            "strong": is_strong,
            "strong_reason": "z" if z_strong else ("rate_p95" if rate_strong else ""),
        }
    return residuals, strong


# ── Spirited-driver guard ──────────────────────────────────────────────
def is_spirited_driver(target: pd.Series, trips_for_imei: pd.DataFrame) -> bool:
    """A vehicle is "spirited" when its trips are dominated by fast cruising.

    Returns True only when ALL of:
      - ≥ SPIRITED_TRIP_FRAC of trips have mean_speed > SPIRITED_MEAN_SPEED_KMH
      - median p90_speed across trips ≥ SPIRITED_MEDIAN_P90_KMH

    The caller is responsible for the n_dual ≤ SPIRITED_MAX_DUAL carve-out
    (we don't demote vehicles that have strong physical evidence even if they
    happen to drive fast).
    """
    if trips_for_imei.empty:
        return False
    mean_speed = pd.to_numeric(trips_for_imei.get("mean_speed"), errors="coerce").dropna()
    p90_speed = pd.to_numeric(trips_for_imei.get("p90_speed"), errors="coerce").dropna()
    if len(mean_speed) == 0 or len(p90_speed) == 0:
        return False
    frac_high_speed = float((mean_speed > C.SPIRITED_MEAN_SPEED_KMH).mean())
    median_p90 = float(p90_speed.median())
    return (frac_high_speed >= C.SPIRITED_TRIP_FRAC
            and median_p90 >= C.SPIRITED_MEDIAN_P90_KMH)


# ── Verdict gate (v5: ML-AND-physics required) ─────────────────────────
def decide_verdict(detector_flag: bool, n_dual: int, n_trips: int,
                   spirited: bool = False) -> tuple[str, str]:
    """v5 verdict gate.

    ELEVATED requires BOTH detector_flag (ML) AND n_dual ≥ ELEVATED_DUAL_WITH_ML.
    Spirited drivers (high-speed dominated trips) are demoted to NORMAL when
    n_dual ≤ SPIRITED_MAX_DUAL — no demotion when physical evidence is strong.

    Returns (verdict, reason). Caller must skip vehicles with n_trips==0
    (validate() does this) — they never reach the verdict gate.
    """
    # Spirited-driver guard fires before any ELEVATED decision when physical
    # evidence is weak. CONFIRMED-class (≥2 strong dual) is never demoted.
    if spirited and n_dual <= C.SPIRITED_MAX_DUAL:
        return "NORMAL", "SPIRITED_DRIVER_HIGH_SPEED_HIGH_RPM"

    if detector_flag and n_dual >= C.ELEVATED_DUAL_WITH_ML:
        return "ELEVATED", "ML_AND_DUAL"
    if not detector_flag and n_dual == 0:
        return "NORMAL", "NO_ML_NO_DUAL"
    if detector_flag and n_dual == 0:
        return "WATCH", "ML_ONLY_PATTERN"
    if detector_flag and n_dual == 1:
        return "WATCH", "ML_PLUS_ONE_DUAL"
    if not detector_flag and n_dual >= 1:
        return "WATCH", "PHYSICS_ONLY_NO_ML"
    return "WATCH", "PARTIAL_EVIDENCE"


def build_reason(strong_metrics: list[str]) -> str:
    """Compact human-readable reason string for ELEVATED records."""
    if not strong_metrics:
        return ""
    labels = [C.METRIC_LABELS.get(m, m) for m in strong_metrics]
    return " | ".join(labels)


# ── Validate one vehicle at a time ─────────────────────────────────────
def validate(stage1: pd.DataFrame, profiles: pd.DataFrame,
             trips: pd.DataFrame | None = None) -> pd.DataFrame:
    profile_lookup = profiles.set_index(["variant", "imei"]).to_dict(orient="index")
    trip_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    if trips is not None and not trips.empty:
        for (variant, imei), g in trips.groupby(["variant", "imei"], sort=False):
            trip_lookup[(variant, imei)] = g
    out: list[dict[str, Any]] = []

    for _, flag in stage1.iterrows():
        variant = flag["variant"]
        imei = flag["imei"]
        ml = bool(flag["detector_flag"])
        target_data = profile_lookup.get((variant, imei))

        if target_data is None:
            # v5: drop vehicles with no valid trips entirely instead of emitting
            # a WATCH/DATA_GAP_NO_VALID_TRIPS row.
            continue

        target = pd.Series(target_data)
        target["variant"] = variant
        target["imei"] = imei

        variant_pool = profiles[profiles["variant"] == variant]
        peers = find_peers(target, variant_pool)
        threshold = C.PHYS_Z_THRESH if ml else C.UNFLAGGED_Z_THRESH

        trips_for_imei = trip_lookup.get((variant, imei), pd.DataFrame())
        spirited = is_spirited_driver(target, trips_for_imei)

        if len(peers) < C.PEER_MIN:
            if spirited:
                verdict, reason = "NORMAL", "SPIRITED_DRIVER_HIGH_SPEED_HIGH_RPM"
            else:
                verdict, reason = ("WATCH" if ml else "NORMAL"), "INSUFFICIENT_PEERS"
            strong_metrics: list[str] = []
            residuals: dict[str, Any] = {}
        else:
            residuals, strong_metrics = score_vehicle(target, peers, threshold)
            verdict, reason = decide_verdict(ml, len(strong_metrics),
                                             int(target["n_trips"]), spirited=spirited)

        # Attach per-trip records for the dashboard table
        trip_cols = ["trip_id", "duration_min", "dist_km", "avg_speed", "mean_speed",
                     "p90_speed", "median_torque", "p90_torque", "median_throttle",
                     "p90_throttle", "low_speed_frac", "soc_start", "soc_end",
                     "soc_consumed", "km_per_soc_pct", "peak_temp", "temp_delta_c",
                     "current_per_speed", "current_per_torque",
                     "current_spike_rate", "high_temp_rate",
                     "mode_0_frac", "mode_1_frac"]
        present = [c for c in trip_cols if c in trips_for_imei.columns]
        trips_records = (trips_for_imei[present]
                          .replace({np.nan: None})
                          .to_dict(orient="records")) if not trips_for_imei.empty else []

        # Vehicle-level mode mix — mean of per-trip fracs so the dashboard
        # can show "this vehicle ran 64% Eco / 36% Thunder".
        m0_avg = float(target.get("mode_0_frac") or 0.0)
        m1_avg = float(target.get("mode_1_frac") or 0.0)
        rec = {
            "imei": imei,
            "variant": variant,
            "detector_flag": ml,
            "isolation_score": flag.get("isolation_score"),
            "cluster_id": flag.get("cluster_id"),
            "verdict": verdict,
            "watch_reason": reason,
            "reason": build_reason(strong_metrics),
            "n_trips": int(target["n_trips"]),
            "n_peers": int(len(peers)),
            "drive_hours": round(float(target.get("drive_hours") or 0.0), 3),
            "charge_hours": round(float(target.get("charge_hours") or 0.0), 3),
            "strong_metric_count": int(len(strong_metrics)),
            "strong_metric_ids": strong_metrics,
            "spirited": bool(spirited),
            "mode_0_frac": round(m0_avg, 4),
            "mode_1_frac": round(m1_avg, 4),
            "trips": trips_records,
        }
        for det_id in C.DUAL_IDS:
            rec[f"{det_id}_events_total"] = int(target.get(f"{det_id}_events_total") or 0)
            rec[f"{det_id}_max_dwell_s"] = round(float(target.get(f"{det_id}_max_dwell_s") or 0.0), 2)
            rec[f"{det_id}_rate"] = round(float(target.get(f"{det_id}_rate") or 0.0), 4)
            r = residuals.get(f"{det_id}_rate", {})
            rec[f"{det_id}_z"] = round(float(r["z"]), 3) if r.get("z") is not None else None
            rec[f"{det_id}_strong"] = bool(r.get("strong", False))
        out.append(rec)
    return pd.DataFrame(out)


# ── Summary + IO ───────────────────────────────────────────────────────
def build_summary(results: pd.DataFrame) -> dict[str, Any]:
    # Vehicles with no valid trips are dropped in validate(); the summary
    # therefore only reports vehicles that survived trip filtering.
    vc = results["verdict"].value_counts().to_dict()
    by_variant = (results.groupby("variant")["verdict"].value_counts()
                  .unstack(fill_value=0).to_dict(orient="index"))
    return {
        "pipeline_version": C.PIPELINE_VERSION,
        "total_validated": int(len(results)),
        "elevated": int((results["verdict"] == "ELEVATED").sum()),
        "watch": int((results["verdict"] == "WATCH").sum()),
        "normal": int((results["verdict"] == "NORMAL").sum()),
        "verdict_counts": {str(k): int(v) for k, v in vc.items()},
        "watch_reason_counts": {str(k): int(v) for k, v in
                                results.get("watch_reason", pd.Series(dtype=object))
                                .fillna("").replace("", "UNSPECIFIED")
                                .value_counts().items()},
        "variant_verdict_counts": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                                   for k, v in by_variant.items()},
        "params": {
            "elevated_dual_with_ml": C.ELEVATED_DUAL_WITH_ML,
            "elevated_dual_no_ml": C.ELEVATED_DUAL_NO_ML,
            "phys_z_thresh": C.PHYS_Z_THRESH,
            "unflagged_z_thresh": C.UNFLAGGED_Z_THRESH,
            "peer_k": C.PEER_K,
            "dwell_default": C.DWELL_SAMPLES,
            "dwell_overrides": dict(C.DETECTOR_DWELL),
            "trip_min_distance_km": C.TRIP_MIN_DISTANCE_KM,
            "trip_min_duration_min": C.TRIP_MIN_DURATION_MIN,
            "dual_detectors": list(C.DUAL_IDS),
            "models_included": list(C.INCLUDE_MODELS) if C.INCLUDE_MODELS else None,
        },
    }


def export(results: pd.DataFrame, summary: dict[str, Any],
           json_path: Path = C.STAGE2_OUT, summary_path: Path = C.SUMMARY_OUT) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    order = {"ELEVATED": 0, "WATCH": 1, "NORMAL": 2}
    sorted_r = results.copy()
    sorted_r["_o"] = sorted_r["verdict"].map(order).fillna(9)
    sorted_r = sorted_r.sort_values(["_o", "strong_metric_count"], ascending=[True, False]).drop(columns="_o")

    payload = {"summary": summary,
               "vehicles": json.loads(sorted_r.to_json(orient="records"))}
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[STAGE-2] wrote {json_path}")
    print(f"[STAGE-2] wrote {summary_path}")


# ── Entry point ────────────────────────────────────────────────────────
def run(csv_path: Path = C.INPUT_CSV,
        stage1_path: Path = C.STAGE1_OUT,
        out_json: Path = C.STAGE2_OUT,
        out_summary: Path = C.SUMMARY_OUT) -> pd.DataFrame:
    print(f"[STAGE-2] loading {stage1_path.name}")
    stage1 = load_stage1(stage1_path)
    print(f"  stage-1 vehicles={len(stage1):,} ml_flags={int(stage1['detector_flag'].sum()):,}")

    print(f"[STAGE-2] loading {csv_path.name}")
    telemetry = load_telemetry(csv_path)
    variant_stats = compute_variant_stats(telemetry)
    trips = build_trip_table(telemetry, variant_stats)
    charges = build_charge_events(telemetry, variant_stats)
    print(f"  valid trips={len(trips):,} vehicles_with_trips={trips['imei'].nunique() if len(trips) else 0:,}")

    profiles = build_profiles(trips, charges)
    results = validate(stage1, profiles, trips=trips)
    summary = build_summary(results)
    export(results, summary, out_json, out_summary)

    vc = results["verdict"].value_counts().to_dict()
    print("[VERDICT] " + ", ".join(f"{k}={v}" for k, v in sorted(vc.items(), key=lambda kv: -kv[1])))
    return results


if __name__ == "__main__":
    run()
