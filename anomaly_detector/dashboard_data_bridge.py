"""dashboard_data_bridge.py — write a fresh dashboard_data.json for the v3 dashboard.

The v3 exporter (`export_v3.py`) merges `dashboard_data.json` (per-variant ML
view) with `anomaly_validation_v3.json` (verdicts) into the v3 HTML. The
legacy upstream that produced `dashboard_data.json` is `analysis.py` / the
slow `process_v10.py` loop. Everything that file actually needs for the
dashboard to render correctly is already in `prod/out/stage1_flags.json`:
the ML scores, severity, cluster id, and per-vehicle counts.

This module reads stage1_flags.json + raw telemetry and emits a complete
dashboard_data.json with the per-signal 1D distributions and signal-pair 2D
heatmaps the dashboard chart panels need to render.

    python -m prod.dashboard_data_bridge                  # full chart build
    python -m prod.dashboard_data_bridge --no-charts      # vehicle list only
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as C
from .stage2_validator import load_telemetry


# ── Histogram bin definitions ───────────────────────────────────────────
# Shared bin edges so per-vehicle bars overlay correctly on the per-variant
# fleet bars. Values are expressed as % of that subset's rows.
SIGNAL_BINS: dict[str, np.ndarray] = {
    "rpm":                      np.arange(0, 7001, 250),
    "torque":                   np.arange(0, 201, 10),
    "speed":                    np.arange(0, 81, 5),
    "throttle":                 np.arange(0, 10.1, 0.5),
    "current":                  np.arange(0, 301, 15),
    "controllerTemperature":    np.arange(0, 101, 5),
    "soc":                      np.arange(0, 101, 5),
    # Mode 0 = Eco, 1 = Thunder — one bar per mode value
    "controllerControllerMode": np.array([-0.5, 0.5, 1.5]),
}

# (x_signal, y_signal, x_edges, y_edges) — keep edges coarse for the heatmap.
HEATMAP_PAIRS: list[tuple[str, str, np.ndarray, np.ndarray]] = [
    ("speed",                 "rpm",      np.arange(0, 81, 10),  np.arange(0, 7001, 700)),
    ("speed",                 "throttle", np.arange(0, 81, 10),  np.arange(0, 10.1, 1.0)),
    ("speed",                 "current",  np.arange(0, 81, 10),  np.arange(0, 301, 30)),
    ("current",               "torque",   np.arange(0, 301, 30), np.arange(0, 201, 20)),
    ("controllerTemperature", "current",  np.arange(0, 101, 10), np.arange(0, 301, 30)),
]


def _labels(edges: np.ndarray) -> list[str]:
    return [f"{edges[i]:g}-{edges[i+1]:g}" for i in range(len(edges) - 1)]


def _hist_pct(values: np.ndarray, edges: np.ndarray) -> list[float]:
    """Return histogram bin densities as percentage of total rows."""
    if len(values) == 0:
        return [0.0] * (len(edges) - 1)
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return [0.0] * len(counts)
    return [round(c * 100.0 / total, 3) for c in counts]


def _heatmap_pct(x: np.ndarray, y: np.ndarray,
                 x_edges: np.ndarray, y_edges: np.ndarray) -> list[list[float]]:
    """Returns 2D histogram as % of total rows, indexed [x_bin][y_bin]."""
    if len(x) == 0 or len(y) == 0:
        return [[0.0] * (len(y_edges) - 1) for _ in range(len(x_edges) - 1)]
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    total = counts.sum()
    if total == 0:
        return counts.tolist()
    return np.round(counts * 100.0 / total, 3).tolist()


def _signal_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.empty(0, dtype="float32")
    s = pd.to_numeric(df[col], errors="coerce").to_numpy()
    s = s[np.isfinite(s)]
    # Controller mode is enumerated: only 0 (Eco) and 1 (Thunder) are real.
    # Anything else is sensor garbage and must not appear in the histogram.
    if col == "controllerControllerMode":
        s = s[(s == 0) | (s == 1)]
    return s


def _build_distributions(df: pd.DataFrame) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for col, edges in SIGNAL_BINS.items():
        out[col] = _hist_pct(_signal_array(df, col), edges)
    return out


def _build_fleet_distributions(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col, edges in SIGNAL_BINS.items():
        out[col] = {"labels": _labels(edges),
                    "data": _hist_pct(_signal_array(df, col), edges)}
    return out


def _build_heatmaps(df: pd.DataFrame, with_meta: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for x_col, y_col, x_edges, y_edges in HEATMAP_PAIRS:
        key = f"{x_col}_vs_{y_col}"
        x = _signal_array(df, x_col)
        y = _signal_array(df, y_col)
        # Align lengths (drop rows where either is non-finite)
        if x_col in df.columns and y_col in df.columns:
            xv = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
            yv = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
            mask = np.isfinite(xv) & np.isfinite(yv)
            x = xv[mask]; y = yv[mask]
        data2d = _heatmap_pct(x, y, x_edges, y_edges)
        if with_meta:
            out[key] = {
                "title": f"{x_col} vs {y_col}",
                "x_labels": _labels(x_edges),
                "y_labels": _labels(y_edges),
                "data": data2d,
            }
        else:
            out[key] = data2d
    return out


ROOT = C.STAGE1_OUT.parent.parent.parent  # Task3/
DEFAULT_STAGE1 = C.STAGE1_OUT
DEFAULT_OUT = ROOT / "dashboard_data.json"


def _vehicle_record(v: dict[str, Any]) -> dict[str, Any]:
    """Map a stage-1 flag record into the v3 vehicle schema."""
    severity = v.get("severity", "NORMAL")
    sev_tier = v.get("severity_tier")
    if sev_tier is None:
        sev_tier = {"HIGH": 2, "MEDIUM": 1, "NORMAL": 0}.get(severity, 0)
    return {
        "imei": str(v.get("imei", "")),
        "cluster_id": int(v.get("cluster_id") or 0),
        "distance_to_center": float(v.get("distance_to_center") or 0.0),
        "cluster_outlier": bool(v.get("cluster_outlier", False)),
        "isolation_score": float(v.get("isolation_score") or 0.0),
        "isolation_outlier": bool(v.get("isolation_outlier", False)),
        "is_anomaly": bool(v.get("is_anomaly", False)),
        "severity": severity,
        "severity_tier": int(sev_tier),
        # Optional blocks the dashboard renders only when present. Empty
        # defaults keep the JS happy without forcing us to recompute them.
        "heatmaps_2d": {},
        "distributions_1d": {},
        "n_records": int(v.get("n_records") or 0),
        "stats": {},
        "explanations": [],
        "timeline": None,
        "variant": v.get("variant", ""),
    }


def build(stage1_path: Path = DEFAULT_STAGE1,
          csv_path: Path = C.INPUT_CSV,
          with_charts: bool = True) -> dict[str, Any]:
    payload = json.loads(stage1_path.read_text())
    flagged_vehicles = payload.get("vehicles", [])

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in flagged_vehicles:
        variant = str(v.get("variant", "")).strip()
        if not variant:
            continue
        by_variant[variant].append(_vehicle_record(v))

    # Chart data: load telemetry once, then compute per-variant fleet hists
    # and per-vehicle hists in one pass over the data.
    fleet_dist: dict[str, dict[str, dict[str, Any]]] = {}
    fleet_hm:   dict[str, dict[str, Any]] = {}
    veh_dist:   dict[tuple[str, str], dict[str, list[float]]] = {}
    veh_hm:     dict[tuple[str, str], dict[str, list[list[float]]]] = {}
    veh_stats:  dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    n_records_lookup: dict[tuple[str, str], int] = {}

    if with_charts:
        print(f"[dashboard-data-bridge] loading telemetry for chart panels ...")
        telemetry = load_telemetry(csv_path)
        print(f"  rows={len(telemetry):,} variants={telemetry['variant'].nunique()}")
        # Drive rows feed the dashboard 1D / 2D panels (mirrors stage-1).
        drive = telemetry[telemetry["vehicleState"] == C.STATE_RUN]
        if drive.empty:
            drive = telemetry
        stat_cols = ("rpm", "torque", "current", "speed", "throttle",
                     "controllerTemperature", "soc", "controllerControllerMode")
        for variant, vdf in drive.groupby("variant", sort=False):
            fleet_dist[variant] = _build_fleet_distributions(vdf)
            fleet_hm[variant]   = _build_heatmaps(vdf, with_meta=True)
            for imei, idf in vdf.groupby("imei", sort=False):
                key = (variant, str(imei))
                veh_dist[key] = _build_distributions(idf)
                veh_hm[key]   = _build_heatmaps(idf, with_meta=False)
                n_records_lookup[key] = len(idf)
                # Per-vehicle stats: mean / p50 / p90 for each signal.
                # Powers the motor efficiency map operating point and any
                # other v.stats consumers in the template.
                vstats: dict[str, dict[str, float]] = {}
                for col in stat_cols:
                    if col not in idf.columns:
                        continue
                    s = pd.to_numeric(idf[col], errors="coerce").dropna()
                    if col == "controllerControllerMode":
                        s = s[s.isin([0, 1])]
                    if len(s) == 0:
                        continue
                    vstats[col] = {
                        "mean": float(s.mean()),
                        "p50":  float(s.quantile(0.50)),
                        "p90":  float(s.quantile(0.90)),
                    }
                veh_stats[key] = vstats
        del drive, telemetry  # let the loader's frame go before json.dumps

    variants_out: dict[str, dict[str, Any]] = {}
    sev_counts_total = {"HIGH": 0, "MEDIUM": 0, "NORMAL": 0}
    total_records = 0
    for variant, vehicles in by_variant.items():
        sev = {"HIGH": 0, "MEDIUM": 0, "NORMAL": 0}
        anom = 0
        records = 0
        # Tally cluster sizes from the per-vehicle records so the dashboard
        # can render "Cluster Cn (k vehicles)" and the cluster outlier threshold.
        cluster_sizes_map: dict[int, int] = {}
        distances: list[float] = []
        for v in vehicles:
            key = (variant, str(v.get("imei", "")))
            if with_charts:
                v["distributions_1d"] = veh_dist.get(key, {})
                v["heatmaps_2d"]      = veh_hm.get(key, {})
                v["stats"]            = veh_stats.get(key, {})
                # Prefer the actual telemetry row count over the stage-1 count;
                # stage-1 only sees driving rows above the min-records gate.
                tele_n = n_records_lookup.get(key)
                if tele_n:
                    v["n_records"] = int(tele_n)
            sev[v["severity"]] = sev.get(v["severity"], 0) + 1
            if v["is_anomaly"]:
                anom += 1
            records += v["n_records"]
            cid = int(v.get("cluster_id") or 0)
            cluster_sizes_map[cid] = cluster_sizes_map.get(cid, 0) + 1
            d = v.get("distance_to_center")
            if isinstance(d, (int, float)) and np.isfinite(d):
                distances.append(float(d))
        for k in sev_counts_total:
            sev_counts_total[k] += sev.get(k, 0)
        total_records += records

        # Stage-1 marks the top 10% by distance as cluster-outliers, so the
        # 90th percentile of distances is the effective outlier threshold.
        if distances:
            distance_threshold = float(np.percentile(distances, 90))
        else:
            distance_threshold = 0.0
        # Indexable list: cluster_sizes[cluster_id] = count
        max_cid = max(cluster_sizes_map) if cluster_sizes_map else -1
        cluster_sizes = [int(cluster_sizes_map.get(i, 0)) for i in range(max_cid + 1)]

        variants_out[variant] = {
            "n_vehicles": len(vehicles),
            "n_records": records,
            "cluster_info": {
                "n_clusters": len(cluster_sizes_map),
                "cluster_sizes": cluster_sizes,
                "distance_threshold": distance_threshold,
            },
            "fleet_heatmaps_2d": fleet_hm.get(variant, {}),
            "fleet_distributions_1d": fleet_dist.get(variant, {}),
            "fleet_stats": {},
            "n_anomalies": anom,
            "n_high_severity": sev.get("HIGH", 0),
            "n_medium_severity": sev.get("MEDIUM", 0),
            "vehicles": vehicles,
        }

    total_vehicles = sum(v["n_vehicles"] for v in variants_out.values())
    total_anomalies = sum(v["n_anomalies"] for v in variants_out.values())

    return {
        "signal_pairs_2d": [],
        "attributes_1d": [],
        "config": {
            "min_records": 50,
            "min_vehicles": 5,
            "iso_contamination": C.STAGE1_IFOREST_CONTAM,
            "timeline_z_thresh": 2.0,
            "explain_z_thresh": 1.3,
        },
        "variants": variants_out,
        "summary": {
            "total_vehicles": total_vehicles,
            "total_anomalies": total_anomalies,
            "severity_counts": sev_counts_total,
            "n_variants": len(variants_out),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "method": "stage1_flags-bridge",
        },
    }


def write(stage1_path: Path = DEFAULT_STAGE1,
          out_path: Path = DEFAULT_OUT,
          csv_path: Path = C.INPUT_CSV,
          with_charts: bool = True) -> Path:
    print(f"[dashboard-data-bridge] reading {stage1_path}")
    data = build(stage1_path, csv_path=csv_path, with_charts=with_charts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, separators=(",", ":")))
    summary = data["summary"]
    variants = data["variants"]
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[dashboard-data-bridge] wrote {out_path} ({size_mb:.2f} MB) — "
          f"{summary['total_vehicles']} vehicles across "
          f"{summary['n_variants']} variants "
          f"(anomalies={summary['total_anomalies']})")
    for k in sorted(variants):
        v = variants[k]
        print(f"  {k}: {v['n_vehicles']} vehicles, "
              f"{v['n_anomalies']} anomalies "
              f"(H={v['n_high_severity']}, M={v['n_medium_severity']})")
    return out_path


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1,
                   help=f"stage-1 flags JSON (default: {DEFAULT_STAGE1})")
    p.add_argument("--csv", type=Path, default=C.INPUT_CSV,
                   help="telemetry CSV (defaults to the prod loader's INPUT_CSV)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output dashboard_data.json (default: {DEFAULT_OUT})")
    p.add_argument("--no-charts", action="store_true",
                   help="skip distribution / heatmap computation (smaller file, empty chart panels)")
    args = p.parse_args(argv)
    write(args.stage1, args.out, csv_path=args.csv, with_charts=not args.no_charts)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
