"""
stage1_detector.py — Stage-1 ML pattern detector (production).

Per-variant KMeans + IsolationForest on 7-signal vehicle feature vectors.
Outputs one record per vehicle: imei, variant, is_anomaly, isolation_score,
distance_to_center, cluster_id, severity.

Stage-1 is intentionally minimal in production. Heatmap dumps, distribution
arrays, per-vehicle explanation panels and timeline plots that bloated the
old `dashboard_data.json` are removed — the only consumer is Stage-2, which
needs the boolean flag plus context fields and nothing else.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config as C


def _pick_k(Xs: np.ndarray, k_max: int, seed: int) -> tuple[int, KMeans]:
    """Pick the best K in [2, k_max] by silhouette score.

    Falls back to k=2 if the data is too small to evaluate higher K. Returns
    the chosen (k, fitted_KMeans) so callers don't refit.
    """
    n = len(Xs)
    upper = max(2, min(k_max, n // 5))   # need enough points per cluster
    if upper == 2:
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xs)
        return 2, km

    best_k, best_score, best_km = 2, -np.inf, None
    for k in range(2, upper + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        # Silhouette is undefined if any cluster ends up empty after fitting
        labels = km.labels_
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score, best_km = k, score, km
    if best_km is None:
        best_km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xs)
        best_k = 2
    return best_k, best_km


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


def _per_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    """7-signal feature vector per (variant, imei). Aggregates over driving rows only."""
    drive = df[df["vehicleState"] == C.STATE_RUN].copy()
    if drive.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (variant, imei), g in drive.groupby(["variant", "imei"], sort=False):
        if len(g) < 50:
            continue
        feat: dict[str, Any] = {"variant": variant, "imei": imei, "n_records": int(len(g))}
        for sig in C.STAGE1_SIGNALS:
            if sig not in g.columns:
                continue
            s = pd.to_numeric(g[sig], errors="coerce").dropna()
            if len(s) == 0:
                continue
            feat[f"{sig}_p50"] = float(s.median())
            feat[f"{sig}_p90"] = float(s.quantile(0.90))
            feat[f"{sig}_iqr"] = float(s.quantile(0.75) - s.quantile(0.25))
        rows.append(feat)
    return pd.DataFrame(rows)


def _score_variant(features: pd.DataFrame) -> pd.DataFrame:
    """Run KMeans + IForest on one variant. Mutates `features` in place with score columns."""
    feat_cols = [c for c in features.columns if c not in ("variant", "imei", "n_records")]
    X = features[feat_cols].astype(float).fillna(features[feat_cols].astype(float).median())
    if len(X) < max(20, C.STAGE1_KMEANS_K_MAX * 2):
        features["cluster_id"] = 0
        features["distance_to_center"] = 0.0
        features["isolation_score"] = 0.0
        features["cluster_outlier"] = False
        features["isolation_outlier"] = False
        features["is_anomaly"] = False
        features["severity"] = "NORMAL"
        features["severity_tier"] = 0
        return features

    Xs = StandardScaler().fit_transform(X.values)

    k, km = _pick_k(Xs, C.STAGE1_KMEANS_K_MAX, C.STAGE1_RANDOM_SEED)
    cluster_id = km.predict(Xs)
    distances = np.linalg.norm(Xs - km.cluster_centers_[cluster_id], axis=1)
    cluster_outlier = distances >= np.percentile(distances, 90)

    iso = IsolationForest(
        contamination=C.STAGE1_IFOREST_CONTAM,
        random_state=C.STAGE1_RANDOM_SEED,
        n_estimators=200,
    ).fit(Xs)
    iso_pred = iso.predict(Xs) == -1
    iso_score = -iso.score_samples(Xs)   # higher = more anomalous

    is_anomaly = cluster_outlier | iso_pred
    severity_tier = (cluster_outlier.astype(int) + iso_pred.astype(int))
    severity = pd.Series(["NORMAL"] * len(features))
    severity[severity_tier == 1] = "MEDIUM"
    severity[severity_tier == 2] = "HIGH"

    features = features.copy()
    features["cluster_id"] = cluster_id.astype(int)
    features["distance_to_center"] = np.round(distances, 4)
    features["isolation_score"] = np.round(iso_score, 4)
    features["cluster_outlier"] = cluster_outlier
    features["isolation_outlier"] = iso_pred
    features["is_anomaly"] = is_anomaly
    features["severity_tier"] = severity_tier.astype(int)
    features["severity"] = severity.values
    return features


def load_telemetry(csv_path: Path) -> pd.DataFrame:
    usecols = ["time", "imei", "rpm", "torque", "controllerTemperature",
               "soc", "speed", "throttle", "current", "vehicleState",
               "controllerControllerMode", "vehicle_model_name", "vehicle_variant_name"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=500_000, low_memory=False):
        chunk = chunk.dropna(subset=["vehicle_model_name", "vehicle_variant_name"])
        if C.INCLUDE_MODELS is not None:
            chunk = chunk[chunk["vehicle_model_name"].isin(C.INCLUDE_MODELS)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        scope = list(C.INCLUDE_MODELS) if C.INCLUDE_MODELS else "any model"
        raise ValueError(f"No rows for {scope} found in {csv_path}.")
    df = pd.concat(parts, ignore_index=True)
    df["imei"] = df["imei"].map(_normalize_imei).astype("string")
    df = df.dropna(subset=["imei"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"])
    df["variant"] = df["vehicle_model_name"].astype(str) + " | " + df["vehicle_variant_name"].astype(str)
    for col in ("rpm", "torque", "controllerTemperature", "soc", "speed",
                "throttle", "current", "vehicleState", "controllerControllerMode"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    soc = df["soc"].dropna()
    if len(soc) and soc.max() <= 1.5:
        df["soc"] *= 100
    return df


def run(csv_path: Path = C.INPUT_CSV, out_path: Path = C.STAGE1_OUT) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[STAGE-1] loading {csv_path.name}")
    telemetry = load_telemetry(csv_path)
    print(f"  rows={len(telemetry):,} vehicles={telemetry['imei'].nunique():,}")

    all_results: list[pd.DataFrame] = []
    for variant, vdf in telemetry.groupby("variant", sort=False):
        feats = _per_vehicle_features(vdf)
        if feats.empty:
            continue
        scored = _score_variant(feats)
        all_results.append(scored)
        n_anom = int(scored["is_anomaly"].sum())
        print(f"  {variant}: vehicles={len(scored):,} anomalies={n_anom:,}")

    flags = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    keep = ["imei", "variant", "is_anomaly", "isolation_score", "distance_to_center",
            "cluster_id", "severity", "severity_tier", "n_records"]
    keep = [c for c in keep if c in flags.columns]
    payload = {
        "pipeline_version": C.PIPELINE_VERSION,
        "stage": "stage1",
        "n_vehicles": int(len(flags)),
        "n_anomalies": int(flags["is_anomaly"].sum()) if "is_anomaly" in flags else 0,
        "vehicles": flags[keep].to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[STAGE-1] wrote {out_path} ({payload['n_anomalies']} anomalies / {payload['n_vehicles']} vehicles)")
    return flags


if __name__ == "__main__":
    run()
