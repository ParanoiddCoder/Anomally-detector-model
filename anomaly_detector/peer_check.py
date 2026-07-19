"""
peer_check.py — single-IMEI peer-comparison HTML report.

For one IMEI: recomputes its trips from telemetry, looks each trip up in
prod/out/fleet_reference.csv by (variant, duration_bin, soc_consumed_bin), and
renders a self-contained HTML with per-trip attribute comparisons.

Usage:
    python -m prod.peer_check <imei>
    python -m prod.peer_check <imei> --out my_report.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .explain import explain_vehicle
from .fleet_reference import DETECTOR_EVENT_ATTRS, FLEET_ATTRIBUTES
from .stage2_validator import (
    _normalize_imei,
    _segment_trips,
    _trip_metrics,
    compute_variant_stats,
    load_telemetry,
)


# Module-level caches so batch callers (e.g. validate_elevated) don't re-read
# the 2.4 GB CSV or re-compute variant stats once per IMEI. The cache is keyed
# by the resolved CSV path so swapping inputs invalidates the entry.
_TELEMETRY_CACHE: dict[str, pd.DataFrame] = {}
_VSTATS_CACHE: dict[str, dict[str, dict[str, float]]] = {}


def _cached_load_telemetry(csv_path: Path) -> pd.DataFrame:
    key = str(Path(csv_path).resolve())
    df = _TELEMETRY_CACHE.get(key)
    if df is None:
        df = load_telemetry(csv_path)
        _TELEMETRY_CACHE[key] = df
    return df


def _cached_variant_stats(csv_path: Path,
                          telemetry: pd.DataFrame) -> dict[str, dict[str, float]]:
    key = str(Path(csv_path).resolve())
    stats = _VSTATS_CACHE.get(key)
    if stats is None:
        stats = compute_variant_stats(telemetry)
        _VSTATS_CACHE[key] = stats
    return stats


# Friendly column-name labels for the report. The table headers already say
# "Vehicle average" / "Fleet median", so the row labels stay plain.
ATTR_LABELS: dict[str, str] = {
    "median_current":  "Current (A)",
    "median_temp":     "Ctrl temp (°C)",
    "median_rpm":      "RPM",
    "median_torque":   "Torque",
    "median_throttle": "Throttle",
    "mean_speed":      "Speed (km/h)",
    "mode_0_frac":     "Eco mode (frac)",
    "mode_1_frac":     "Thunder mode (frac)",
}

# Attributes shown in the peer-check table. Current + temperature are the
# primary safety signals; rpm/torque/throttle/speed are included for context.
# mode_0/mode_1 fracs show operating-mode mix vs peers.
DANGER_ATTRIBUTES: tuple[str, ...] = (
    "median_current",
    "median_temp",
    "median_rpm",
    "median_torque",
    "median_throttle",
    "mean_speed",
    "mode_0_frac",
    "mode_1_frac",
)

# Detectors involving current or temperature — the rest (drivetrain, demand
# patterns, drag) are diagnostic, not danger.
DANGER_DETECTORS: tuple[str, ...] = (
    "d3_throttle_current_events",   # throttle high, current low — controller fault
    "d4_current_torque_events",     # current high, torque low — motor inefficiency
    "d5_current_rpm_torque_events", # current+rpm high, torque low — driveline
    "d8_temp_overheat_events",      # controller temp high, current low — cooling fault
)

# Attributes where higher = worse (for colour coding).
HIGHER_IS_WORSE = {
    "median_current", "p90_current",
    "median_temp", "peak_temp",
}

# Map from detector-event column name → human label (built from C.DUAL_DETECTORS).
DETECTOR_EVENT_LABELS: dict[str, str] = {
    f"{d['id']}_events": d["label"] for d in C.DUAL_DETECTORS
}


# ── Bin helpers (must match prod/fleet_reference.py exactly) ───────────
def _bin(value: float, width: int) -> int:
    return int(round(value / width)) * width


# ── Loaders ────────────────────────────────────────────────────────────
def _load_fleet_ref(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["variant"] = df["variant"].astype(str).str.strip()
    return df


def _load_stage2_for_imei(imei: str, path: Path) -> dict[str, Any] | None:
    """Read prod/out/stage2_verdicts.json and find this imei (best-effort)."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    for r in payload.get("vehicles", []):
        if str(r.get("imei")) == str(imei):
            return r
    return None


def _compute_trips_for_imei(
    imei: str, csv_path: Path
) -> tuple[str | None, list[dict[str, Any]], dict[int, pd.DataFrame]]:
    """Recompute trip metrics from raw telemetry for a single IMEI.

    Returns (variant, [trip_metric_dict, ...], {trip_id: raw_trip_rows_df}).
    The raw rows are kept so we can build per-attribute time-in-bucket
    histograms downstream. variant=None if imei not found.
    """
    cached = str(Path(csv_path).resolve()) in _TELEMETRY_CACHE
    if not cached:
        print(f"[peer_check] loading telemetry from {csv_path.name} ...")
    telemetry = _cached_load_telemetry(csv_path)
    target = _normalize_imei(imei)
    veh = telemetry[telemetry["imei"] == target]
    if veh.empty:
        return None, [], {}
    variant = str(veh["variant"].iloc[0])
    vstats_all = _cached_variant_stats(csv_path, telemetry)
    vstats = vstats_all.get(variant, {})

    active = _segment_trips(veh)
    if active.empty:
        return variant, [], {}
    out: list[dict[str, Any]] = []
    raw_by_trip: dict[int, pd.DataFrame] = {}
    for trip_id, trip in active.groupby("trip_id", sort=True):
        m = _trip_metrics(trip, vstats)
        if m is None:
            continue
        m["trip_id"] = int(trip_id)
        out.append(m)
        raw_by_trip[int(trip_id)] = trip.reset_index(drop=True)
    return variant, out, raw_by_trip


# ── Comparison core ────────────────────────────────────────────────────
def _bucket_lookup(fleet: pd.DataFrame, variant: str,
                   duration_min: float, soc_consumed: float) -> pd.Series | None:
    dbin = _bin(duration_min, C.FLEET_DURATION_BIN_MIN)
    sbin = _bin(soc_consumed, C.FLEET_SOC_BIN_PCT)
    hit = fleet[(fleet["variant"] == variant)
                & (fleet["duration_bin"] == dbin)
                & (fleet["soc_consumed_bin"] == sbin)]
    if not hit.empty:
        return hit.iloc[0]
    # Fall-back: nearest bucket within same variant (Manhattan distance on bin grid).
    same_v = fleet[fleet["variant"] == variant]
    if same_v.empty:
        return None
    dists = (same_v["duration_bin"] - dbin).abs() + (same_v["soc_consumed_bin"] - sbin).abs()
    return same_v.iloc[int(dists.values.argmin())]


def _compare_trip(trip: dict[str, Any], bucket: pd.Series | None) -> list[dict[str, Any]]:
    """Produce one comparison row per attribute for this trip.

    Only DANGER_ATTRIBUTES (current + temperature) are surfaced. Other
    attributes are diagnostic, not safety alerts, and are suppressed.
    """
    rows: list[dict[str, Any]] = []
    danger_set = set(DANGER_ATTRIBUTES)
    for attr in FLEET_ATTRIBUTES:
        if attr not in danger_set:
            continue
        v = trip.get(attr)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if bucket is None:
            rows.append({"attr": attr, "vehicle": float(v),
                         "fleet": float("nan"), "delta": float("nan"),
                         "pct_dev": float("nan"), "flag": "no-bucket"})
            continue
        fcol = f"median_{attr}"
        f = bucket.get(fcol, float("nan"))
        f = float(f) if pd.notna(f) else float("nan")
        delta = float(v) - f if np.isfinite(f) else float("nan")
        pct = (delta / f * 100.0) if (np.isfinite(f) and f != 0) else float("nan")
        # Threshold for the colour flag — ±25% deviation is noteworthy.
        if not np.isfinite(pct):
            flag = "n/a"
        elif abs(pct) < 15:
            flag = "ok"
        elif abs(pct) < 30:
            flag = "warn"
        else:
            flag = "alert"
        rows.append({"attr": attr, "vehicle": float(v), "fleet": f,
                     "delta": delta, "pct_dev": pct, "flag": flag})
    return rows


def _detector_status(events: float, fleet_med: float) -> str:
    """Status logic for integer detector firings (different from continuous %dev).

    - 0 events             → 'ok' (didn't fire)
    - fired AND fleet=0    → 'alert' (peers in this bucket don't fire here)
    - fired AND >2× fleet  → 'alert'
    - fired AND >fleet     → 'warn'
    - fired AND ≤fleet     → 'warn' (fired but in line with peers)
    """
    if not np.isfinite(events) or events <= 0:
        return "ok"
    if not np.isfinite(fleet_med) or fleet_med <= 0:
        return "alert"
    if events > 2 * fleet_med:
        return "alert"
    return "warn"


def _compare_detector_events(trip: dict[str, Any],
                             bucket: pd.Series | None) -> list[dict[str, Any]]:
    """Produce one row per dual detector for this trip — events + max dwell vs fleet.

    Filtered to DANGER_DETECTORS (those involving current or temperature).
    """
    rows: list[dict[str, Any]] = []
    danger_set = set(DANGER_DETECTORS)
    for attr in DETECTOR_EVENT_ATTRS:
        if attr not in danger_set:
            continue
        det_id = attr.replace("_events", "")
        events = trip.get(attr, 0) or 0
        max_dwell = trip.get(f"{det_id}_max_dwell_s", 0.0) or 0.0
        if bucket is None:
            fleet_med = float("nan")
        else:
            f = bucket.get(f"median_{attr}", float("nan"))
            fleet_med = float(f) if pd.notna(f) else float("nan")
        status = _detector_status(float(events), fleet_med)
        rows.append({
            "attr": attr,
            "det_id": det_id,
            "label": DETECTOR_EVENT_LABELS.get(attr, det_id),
            "events": int(events),
            "max_dwell_s": float(max_dwell),
            "fleet_median": fleet_med,
            "flag": status,
        })
    return rows


def _detector_rollup(per_trip_detectors: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Vehicle-level rollup across all trips: total events vs sum of fleet medians."""
    if not per_trip_detectors:
        return []
    by_attr: dict[str, dict[str, Any]] = {}
    for rows in per_trip_detectors:
        for r in rows:
            entry = by_attr.setdefault(r["attr"], {
                "attr": r["attr"],
                "det_id": r["det_id"],
                "label": r["label"],
                "events_total": 0,
                "max_dwell_s": 0.0,
                "fleet_total": 0.0,
                "fleet_buckets_n": 0,
                "trips_fired": 0,
            })
            entry["events_total"] += int(r["events"])
            entry["max_dwell_s"] = max(entry["max_dwell_s"], float(r["max_dwell_s"]))
            if np.isfinite(r["fleet_median"]):
                entry["fleet_total"] += float(r["fleet_median"])
                entry["fleet_buckets_n"] += 1
            if r["events"] > 0:
                entry["trips_fired"] += 1
    out: list[dict[str, Any]] = []
    for entry in by_attr.values():
        flag = _detector_status(entry["events_total"], entry["fleet_total"])
        entry["flag"] = flag
        out.append(entry)
    out.sort(key=lambda e: -int(e["events_total"]))
    return out


# ── Plotting ───────────────────────────────────────────────────────────
def _trip_chart_png(rows: list[dict[str, Any]], title: str) -> str:
    """Horizontal bar chart of %deviation per attribute. Returns base64 PNG."""
    rows = [r for r in rows if np.isfinite(r["pct_dev"])]
    if not rows:
        return ""
    labels = [ATTR_LABELS.get(r["attr"], r["attr"]) for r in rows]
    vals = [r["pct_dev"] for r in rows]
    colors = []
    for r in rows:
        if r["flag"] == "alert":
            colors.append("#d62728")
        elif r["flag"] == "warn":
            colors.append("#ff9f1c")
        else:
            colors.append("#2ca02c")
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(rows) + 1.0))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.axvline(15, color="grey", linewidth=0.4, linestyle="--")
    ax.axvline(-15, color="grey", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Deviation from fleet median (%)", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _summary_chart_png(per_trip: list[list[dict[str, Any]]]) -> str:
    """Average %deviation per attribute across all trips."""
    agg: dict[str, list[float]] = {}
    for rows in per_trip:
        for r in rows:
            if np.isfinite(r["pct_dev"]):
                agg.setdefault(r["attr"], []).append(r["pct_dev"])
    if not agg:
        return ""
    items = [(a, float(np.mean(vs))) for a, vs in agg.items()]
    items.sort(key=lambda x: -abs(x[1]))
    labels = [ATTR_LABELS.get(a, a) for a, _ in items]
    vals = [v for _, v in items]
    colors = ["#d62728" if abs(v) >= 30 else ("#ff9f1c" if abs(v) >= 15 else "#2ca02c")
              for v in vals]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(items) + 1.0))
    y = np.arange(len(items))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Mean deviation across trips (%)", fontsize=10)
    ax.set_title("Per-attribute average deviation from fleet", fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── HTML render ────────────────────────────────────────────────────────
_HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>Peer Check — {imei}</title>
<style>
 body {{ font-family:-apple-system, system-ui, sans-serif; margin:0; padding:24px;
        background:#0e1014; color:#e6e8ee; }}
 h1 {{ font-size:22px; margin:0 0 6px 0; }}
 h2 {{ font-size:16px; margin:24px 0 8px 0; color:#9ab; }}
 h3 {{ font-size:14px; margin:18px 0 6px 0; color:#cde; }}
 .card {{ background:#181c24; border:1px solid #2a2f3a; border-radius:8px;
          padding:16px; margin-bottom:14px; }}
 .meta span {{ display:inline-block; margin-right:18px; color:#9ab; font-size:13px; }}
 .meta b {{ color:#e6e8ee; }}
 .verdict {{ display:inline-block; padding:2px 10px; border-radius:12px;
            font-size:12px; font-weight:600; }}
 .v-ELEVATED {{ background:#7a1f1f; color:#fff; }}
 .v-WATCH    {{ background:#7a5a1f; color:#fff; }}
 .v-NORMAL   {{ background:#1f5a3a; color:#fff; }}
 .v-NA       {{ background:#404552; color:#fff; }}
 table {{ border-collapse:collapse; width:100%; font-size:12px; }}
 th, td {{ border-bottom:1px solid #2a2f3a; padding:5px 8px; text-align:right; }}
 th {{ background:#1f2532; text-align:left; }}
 td.attr {{ text-align:left; color:#cde; }}
 tr.ok      td.flag {{ color:#3ad29f; }}
 tr.warn    td.flag {{ color:#f6b042; }}
 tr.alert   td.flag {{ color:#ff5b6a; font-weight:600; }}
 tr.no-bucket td.flag, tr.na td.flag {{ color:#666; }}
 .fired-alert {{ display:inline-block; background:#7a1f1f; color:#fff;
                 padding:2px 8px; border-radius:10px; font-size:11px;
                 margin-right:6px; font-family:'IBM Plex Mono',monospace; }}
 .fired-warn  {{ display:inline-block; background:#7a5a1f; color:#fff;
                 padding:2px 8px; border-radius:10px; font-size:11px;
                 margin-right:6px; font-family:'IBM Plex Mono',monospace; }}
 .fired-none  {{ color:#667; font-size:12px; font-style:italic; }}
 .section-label {{ color:#9ab; font-size:11px; text-transform:uppercase;
                   letter-spacing:.05em; margin:14px 0 4px 0; }}
 img.chart {{ display:block; margin:10px 0; max-width:100%;
              background:#fff; border-radius:6px; }}
 .explain ul {{ margin:6px 0 0 0; padding-left:20px; }}
 .explain li {{ margin-bottom:10px; line-height:1.5; font-size:13px; color:#dde; }}
 .grid {{ display:grid; grid-template-columns:1.1fr 0.9fr; gap:14px; }}
 .footer {{ color:#667; font-size:11px; margin-top:30px; }}
</style></head><body>
"""


def _verdict_class(v: str | None) -> str:
    return f"v-{v}" if v in {"ELEVATED", "WATCH", "NORMAL"} else "v-NA"


def _fmt(x: Any) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:,.2f}"
    return str(x)


def _trip_table_html(rows: list[dict[str, Any]]) -> str:
    out = ["<table><tr><th>Attribute</th><th>Vehicle</th><th>Fleet median</th>"
           "<th>Δ</th><th>%dev</th><th>Flag</th></tr>"]
    for r in rows:
        out.append(
            f"<tr class='{r['flag']}'>"
            f"<td class='attr'>{ATTR_LABELS.get(r['attr'], r['attr'])}</td>"
            f"<td>{_fmt(r['vehicle'])}</td>"
            f"<td>{_fmt(r['fleet'])}</td>"
            f"<td>{_fmt(r['delta'])}</td>"
            f"<td>{_fmt(r['pct_dev'])}</td>"
            f"<td class='flag'>{r['flag']}</td>"
            f"</tr>"
        )
    out.append("</table>")
    return "".join(out)


def _detector_table_html(rows: list[dict[str, Any]]) -> str:
    """Per-trip detector firings — one row per dual detector."""
    out = ["<table><tr><th>Detector</th><th>Events (this trip)</th>"
           "<th>Max dwell (s)</th><th>Fleet median events</th><th>Status</th></tr>"]
    for r in rows:
        out.append(
            f"<tr class='{r['flag']}'>"
            f"<td class='attr'>{r['label']}</td>"
            f"<td>{r['events']}</td>"
            f"<td>{_fmt(r['max_dwell_s'])}</td>"
            f"<td>{_fmt(r['fleet_median'])}</td>"
            f"<td class='flag'>{r['flag']}</td>"
            f"</tr>"
        )
    out.append("</table>")
    return "".join(out)


def _rollup_table_html(rows: list[dict[str, Any]], n_trips: int) -> str:
    """Vehicle-wide detector rollup — totals across all trips vs fleet sum."""
    out = ["<table><tr><th>Detector</th><th>Total events</th>"
           "<th>Trips fired</th><th>Max dwell (s)</th>"
           "<th>Fleet total (sum of bucket medians)</th><th>Status</th></tr>"]
    for r in rows:
        out.append(
            f"<tr class='{r['flag']}'>"
            f"<td class='attr'>{r['label']}</td>"
            f"<td>{r['events_total']}</td>"
            f"<td>{r['trips_fired']}/{n_trips}</td>"
            f"<td>{_fmt(r['max_dwell_s'])}</td>"
            f"<td>{_fmt(r['fleet_total'])}</td>"
            f"<td class='flag'>{r['flag']}</td>"
            f"</tr>"
        )
    out.append("</table>")
    return "".join(out)


def _fired_badge(rows: list[dict[str, Any]]) -> str:
    """Inline badge listing detectors that fired on this trip."""
    fired = [r for r in rows if r["events"] > 0]
    if not fired:
        return "<span class='fired-none'>no detectors fired</span>"
    parts = []
    for r in fired:
        cls = "fired-alert" if r["flag"] == "alert" else "fired-warn"
        parts.append(f"<span class='{cls}'>{r['det_id']}: {r['events']}</span>")
    return " ".join(parts)


def _vehicle_attr_averages(trips: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of each danger attribute across all of this vehicle's trips."""
    out: dict[str, float] = {}
    for attr in DANGER_ATTRIBUTES:
        vals = [t[attr] for t in trips
                if isinstance(t.get(attr), (int, float)) and np.isfinite(t[attr])]
        out[attr] = float(np.mean(vals)) if vals else float("nan")
    return out


def _count_elevated_trips(trips: list[dict[str, Any]]) -> int:
    """A trip is 'elevated' when ≥1 danger detector (D3/D4/D5/D8) fired on it."""
    danger_event_keys = [a for a in DANGER_DETECTORS]   # e.g. d3_throttle_current_events
    n = 0
    for t in trips:
        if any(int(t.get(k, 0) or 0) > 0 for k in danger_event_keys):
            n += 1
    return n


def _is_trip_elevated(trip_metrics: dict[str, Any]) -> bool:
    return any(int(trip_metrics.get(k, 0) or 0) > 0 for k in DANGER_DETECTORS)


# Telemetry-column buckets for the time-in-bucket section. Each entry is
# (telemetry_column, label, list_of_bucket_edges).
BUCKET_DEFS: list[tuple[str, str, list[float]]] = [
    ("current",                "Current (A)",      [0, 50, 100, 150, 200, float("inf")]),
    ("controllerTemperature",  "Ctrl temp (°C)",   [0, 25, 40, 55, 70, float("inf")]),
    ("rpm",                    "RPM",              [0, 1500, 3000, 4500, 6000, float("inf")]),
    ("torque",                 "Torque",           [0, 25, 50, 75, 100, float("inf")]),
    ("throttle",               "Throttle",         [0, 2, 4, 6, 8, float("inf")]),
    ("speed",                  "Speed (km/h)",     [0, 15, 30, 45, 60, float("inf")]),
]

# Bin edges for 2D heatmaps — coarse enough to be readable as a small PNG.
HEATMAP_BINS: dict[str, np.ndarray] = {
    "current":                np.array([0, 50, 100, 150, 200, 250, 300]),
    "rpm":                    np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]),
    "torque":                 np.array([0, 25, 50, 75, 100, 125, 150]),
    "throttle":               np.array([0, 2, 4, 6, 8, 10]),
    "speed":                  np.array([0, 10, 20, 30, 40, 50, 60, 80]),
    "controllerTemperature":  np.array([0, 20, 30, 40, 50, 60, 80]),
}

# Signal pairs for the heatmap section. (x_col, y_col, x_label, y_label).
HEATMAP_PAIRS: list[tuple[str, str, str, str]] = [
    ("current", "rpm",                    "Current (A)", "RPM"),
    ("current", "torque",                 "Current (A)", "Torque"),
    ("current", "throttle",               "Current (A)", "Throttle"),
    ("current", "speed",                  "Current (A)", "Speed (km/h)"),
    ("current", "controllerTemperature",  "Current (A)", "Ctrl temp (°C)"),
    ("rpm",     "torque",                 "RPM",          "Torque"),
    ("rpm",     "speed",                  "RPM",          "Speed (km/h)"),
    ("torque",  "speed",                  "Torque",       "Speed (km/h)"),
    ("throttle","speed",                  "Throttle",     "Speed (km/h)"),
    ("controllerTemperature", "speed",    "Ctrl temp (°C)", "Speed (km/h)"),
]


def _bucket_label(edges: list[float], i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    return f"{lo:g}+" if hi == float("inf") else f"{lo:g}–{hi:g}"


def _elevated_trip_rows(
    trips: list[dict[str, Any]],
    raw_by_trip: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate raw telemetry rows from this vehicle's elevated trips only."""
    frames = [raw_by_trip[int(t["trip_id"])] for t in trips
              if _is_trip_elevated(t) and int(t["trip_id"]) in raw_by_trip]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _all_trip_rows(
    trips: list[dict[str, Any]],
    raw_by_trip: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate raw telemetry rows from every valid trip of this vehicle."""
    frames = [raw_by_trip[int(t["trip_id"])] for t in trips
              if int(t["trip_id"]) in raw_by_trip]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fleet_drive_rows(csv_path: Path, variant: str) -> pd.DataFrame:
    """All driving rows (state==2, speed>0) for the variant, across the fleet."""
    tel = _cached_load_telemetry(csv_path)
    return tel[(tel["variant"] == variant)
               & (tel["vehicleState"] == C.STATE_RUN)
               & (tel["speed"].fillna(0) > 0)]


def _pct_in_buckets(rows: pd.DataFrame, col: str,
                    edges: list[float]) -> list[float]:
    """Percent of rows falling into each [lo, hi) bucket of `col`."""
    if rows.empty or col not in rows.columns:
        return [0.0] * (len(edges) - 1)
    s = pd.to_numeric(rows[col], errors="coerce").dropna()
    total = len(s)
    if total == 0:
        return [0.0] * (len(edges) - 1)
    out: list[float] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        out.append(round(float(((s >= lo) & (s < hi)).sum()) / total * 100.0, 2))
    return out


def _heatmap_png(panels: list[tuple[pd.DataFrame, str]],
                 x_col: str, y_col: str,
                 x_label: str, y_label: str) -> str:
    """N-panel 2D histogram PNG (base64). Each panel is (rows_df, title)."""
    x_bins = HEATMAP_BINS.get(x_col)
    y_bins = HEATMAP_BINS.get(y_col)
    if x_bins is None or y_bins is None or not panels:
        return ""

    def _hist(rows: pd.DataFrame) -> np.ndarray:
        if rows.empty or x_col not in rows.columns or y_col not in rows.columns:
            return np.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype=float)
        x = pd.to_numeric(rows[x_col], errors="coerce")
        y = pd.to_numeric(rows[y_col], errors="coerce")
        ok = x.notna() & y.notna()
        if not ok.any():
            return np.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype=float)
        h, _, _ = np.histogram2d(x[ok].to_numpy(), y[ok].to_numpy(),
                                 bins=[x_bins, y_bins])
        # As %-of-rows so panels with different sizes are comparable.
        total = h.sum()
        if total > 0:
            h = h / total * 100.0
        return h.T   # transpose so y is vertical for imshow

    hists = [(_hist(rows), title) for rows, title in panels]
    vmax = max((float(h.max()) for h, _ in hists), default=1e-9)
    vmax = max(vmax, 1e-9)

    n = len(hists)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.6, 3.6),
                             squeeze=False)
    for ax, (h, title) in zip(axes[0], hists):
        im = ax.imshow(h, origin="lower", aspect="auto", cmap="magma",
                       vmin=0, vmax=vmax,
                       extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.tick_params(labelsize=8)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("% of rows", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Range (km per SOC %) comparison ────────────────────────────────────
def _vehicle_range(trips: list[dict[str, Any]], elevated: bool) -> float:
    vals = [t.get("km_per_soc_pct") for t in trips
            if _is_trip_elevated(t) == elevated]
    vals = [float(v) for v in vals
            if isinstance(v, (int, float)) and np.isfinite(v) and v > 0]
    return float(np.mean(vals)) if vals else float("nan")


def _fleet_range_median(stage2_path: Path, variant: str) -> float:
    """Median km_per_soc_pct across all valid trips in the variant.

    Reads from stage2_verdicts.json because the per-trip range value is
    stored on each vehicle's `trips` array — the fleet_reference.csv only
    has bucket aggregates and does not include this column.
    """
    if not stage2_path.exists():
        return float("nan")
    try:
        payload = json.loads(stage2_path.read_text())
    except Exception:
        return float("nan")
    vals: list[float] = []
    for v in payload.get("vehicles", []):
        if v.get("variant") != variant:
            continue
        for t in v.get("trips") or []:
            kps = t.get("km_per_soc_pct")
            if isinstance(kps, (int, float)) and np.isfinite(kps) and kps > 0:
                vals.append(float(kps))
    return float(np.median(vals)) if vals else float("nan")


def _fleet_attr_averages(fleet: pd.DataFrame, variant: str) -> dict[str, float]:
    """Fleet median for each danger attribute (median across all variant buckets)."""
    out: dict[str, float] = {}
    sub = fleet[fleet["variant"] == variant]
    for attr in DANGER_ATTRIBUTES:
        col = f"median_{attr}"
        if col in sub.columns:
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            out[attr] = float(s.median()) if len(s) else float("nan")
        else:
            out[attr] = float("nan")
    return out


def render_html(imei: str, variant: str, stage2: dict[str, Any] | None,
                trips: list[dict[str, Any]],
                per_trip_rows: list[list[dict[str, Any]]],
                per_trip_detectors: list[list[dict[str, Any]]],
                fleet_path: Path,
                fleet_df: pd.DataFrame | None = None,
                raw_by_trip: dict[int, pd.DataFrame] | None = None,
                stage2_path: Path | None = None,
                csv_path: Path | None = None) -> str:
    verdict = (stage2 or {}).get("verdict", "—")

    n_elev = _count_elevated_trips(trips)
    n_norm = len(trips) - n_elev
    parts: list[str] = [_HTML_HEAD.format(imei=imei)]
    parts.append(f"<h1>Peer-comparison report — IMEI {imei}</h1>")
    parts.append("<div class='card meta'>")
    parts.append(f"<span>Variant: <b>{variant}</b></span>")
    parts.append(f"<span>Verdict: <span class='verdict {_verdict_class(verdict)}'>{verdict}</span></span>")
    parts.append(f"<span>Elevated trips: <b>{n_elev} / {len(trips)}</b></span>")
    parts.append("</div>")

    # ── Filter bar (controller mode + verdict) ─────────────────────────
    parts.append(
        "<div class='card' style='padding:10px 16px'>"
        "<div style='display:flex;gap:32px;flex-wrap:wrap;align-items:center;font-size:13px'>"
        "  <div><span style='color:#9ab;margin-right:8px'>Controller mode:</span>"
        "    <label style='margin-right:10px'><input type='radio' name='modeFlt' value='all' checked> Combined</label>"
        "    <label style='margin-right:10px'><input type='radio' name='modeFlt' value='eco'> Eco (0)</label>"
        "    <label><input type='radio' name='modeFlt' value='thunder'> Thunder (1)</label>"
        "  </div>"
        "  <div><span style='color:#9ab;margin-right:8px'>Verdict:</span>"
        "    <label style='margin-right:10px'><input type='radio' name='vrdFlt' value='all' checked> All</label>"
        "    <label style='margin-right:10px'><input type='radio' name='vrdFlt' value='1'> Elevated</label>"
        "    <label><input type='radio' name='vrdFlt' value='0'> Normal</label>"
        "  </div>"
        "  <div style='color:#9ab;margin-left:auto'>Showing <b id='visTrips' style='color:#e6e8ee'>0</b> / "
        "<span id='totTrips'>0</span> trips</div>"
        "</div></div>"
    )

    # ── Why the model tagged this vehicle ──────────────────────────────
    if stage2:
        bullets = explain_vehicle(stage2)
        if bullets:
            parts.append("<div class='card explain'>")
            parts.append("<h2>Why this vehicle was tagged</h2>")
            parts.append("<div style='color:#9ab;font-size:12px;margin-bottom:8px'>"
                         "Stage-1 ML (KMeans + IsolationForest on per-vehicle "
                         "usage features) flags an unusual driving pattern. "
                         "Stage-2 then checks 8 dual-signal physics detectors "
                         "(e.g. current-vs-torque, throttle-vs-current) on a "
                         "per-trip basis. The verdict gate combines both: "
                         "ELEVATED needs ML <em>and</em> ≥2 strong physics "
                         "detectors; WATCH covers partial evidence; NORMAL is "
                         "everything else.</div>")
            parts.append("<ul>")
            for b in bullets:
                html_b = b.replace("\n", "<br>").lstrip("• ")
                parts.append(f"<li>{html_b}</li>")
            parts.append("</ul></div>")

    # Single comparison table: vehicle average across trips vs fleet median.
    if fleet_df is None:
        fleet_df = _load_fleet_ref(fleet_path)
    veh_avg = _vehicle_attr_averages(trips)
    flt_avg = _fleet_attr_averages(fleet_df, variant)

    parts.append("<div class='card'>")
    parts.append("<h2>Attribute averages — vehicle vs fleet "
                 "<span style='color:#667;font-size:11px;font-weight:normal'>"
                 "(vehicle average recomputes when you change filters)</span></h2>")
    parts.append("<table id='attrAvgTable'><tr><th>Attribute</th><th>Vehicle average</th>"
                 "<th>Fleet median</th></tr>")
    for attr in DANGER_ATTRIBUTES:
        parts.append(
            f"<tr data-attr='{attr}'><td class='attr'>{ATTR_LABELS.get(attr, attr)}</td>"
            f"<td class='veh-avg'>{_fmt(veh_avg.get(attr))}</td>"
            f"<td>{_fmt(flt_avg.get(attr))}</td></tr>"
        )
    parts.append("</table></div>")

    # ── Raw elevated trips table ───────────────────────────────────────
    def _raw_trip_table(title: str, blurb: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        parts.append("<div class='card'>")
        parts.append(f"<h2>{title}</h2>")
        parts.append(f"<div style='color:#9ab;font-size:12px;margin-bottom:8px'>{blurb}</div>")
        parts.append("<table><tr><th>Trip ID</th><th>Duration (min)</th>"
                     "<th>Distance (km)</th><th>SOC consumed (%)</th>"
                     "<th>Range (km/%)</th><th>Median current (A)</th>"
                     "<th>Peak temp (°C)</th><th>Mode (Eco / Thunder)</th></tr>")
        elev_tag = " <span class='fired-warn'>elev</span>"
        for t in rows:
            is_elev = _is_trip_elevated(t)
            tag = elev_tag if is_elev else ""
            m0 = t.get("mode_0_frac") or 0.0
            m1 = t.get("mode_1_frac") or 0.0
            mode_cls = "thunder" if m1 >= m0 else "eco"
            elev_attr = "1" if is_elev else "0"
            mode_bar = (
                f"<div style='display:flex;height:14px;width:120px;border-radius:3px;overflow:hidden'>"
                f"<div title='Eco {m0:.0%}' style='flex:{m0:.4f};background:#4caf50'></div>"
                f"<div title='Thunder {m1:.0%}' style='flex:{m1:.4f};background:#f44336'></div>"
                f"</div>"
                f"<div style='font-size:10px;color:#9ab'>"
                f"Eco:{m0:.0%} Thunder:{m1:.0%}</div>"
            )
            parts.append(
                f"<tr class='trip-row' data-mode='{mode_cls}' data-elev='{elev_attr}'>"
                f"<td>{t.get('trip_id', '—')}{tag}</td>"
                f"<td>{_fmt(t.get('duration_min'))}</td>"
                f"<td>{_fmt(t.get('dist_km'))}</td>"
                f"<td>{_fmt(t.get('soc_consumed'))}</td>"
                f"<td>{_fmt(t.get('km_per_soc_pct'))}</td>"
                f"<td>{_fmt(t.get('median_current'))}</td>"
                f"<td>{_fmt(t.get('peak_temp'))}</td>"
                f"<td>{mode_bar}</td></tr>"
            )
        parts.append("</table></div>")

    elev_trips = [t for t in trips if _is_trip_elevated(t)]
    _raw_trip_table(
        "Raw elevated trips",
        "Per-trip detail for the elevated trips of this vehicle. "
        "Range = trip distance ÷ SOC consumed.",
        elev_trips,
    )
    _raw_trip_table(
        "Raw — all trips",
        "Per-trip detail for every valid trip of this vehicle. "
        "Elevated trips are flagged in the Trip ID column.",
        trips,
    )

    # ── Range comparison: all trips, elevated trips, fleet ─────────────
    fleet_range = _fleet_range_median(
        stage2_path if stage2_path is not None else C.STAGE2_OUT, variant
    )
    veh_elev_range = _vehicle_range(trips, elevated=True)
    # All-trips mean: average km_per_soc_pct across every valid trip.
    all_kps = [t.get("km_per_soc_pct") for t in trips]
    all_kps = [float(v) for v in all_kps
               if isinstance(v, (int, float)) and np.isfinite(v) and v > 0]
    veh_all_range = float(np.mean(all_kps)) if all_kps else float("nan")

    parts.append("<div class='card'>")
    parts.append("<h2>Range — vehicle vs fleet</h2>")
    parts.append("<table><tr><th>Source</th><th>Trips</th>"
                 "<th>Distance ÷ SOC consumed (km / %)</th></tr>")
    parts.append(
        f"<tr><td class='attr'>Vehicle — all trips (mean)</td>"
        f"<td>{len(trips)}</td><td>{_fmt(veh_all_range)}</td></tr>"
    )
    parts.append(
        f"<tr><td class='attr'>Vehicle — elevated trips (mean)</td>"
        f"<td>{n_elev}</td><td>{_fmt(veh_elev_range)}</td></tr>"
    )
    parts.append(
        f"<tr><td class='attr'>Fleet — {variant} (median across all trips)</td>"
        f"<td>—</td><td>{_fmt(fleet_range)}</td></tr>"
    )
    parts.append("</table></div>")

    # ── Time-in-bucket: elevated trips vs fleet ────────────────────────
    elev_rows = pd.DataFrame()
    all_rows = pd.DataFrame()
    if raw_by_trip:
        elev_rows = _elevated_trip_rows(trips, raw_by_trip)
        all_rows = _all_trip_rows(trips, raw_by_trip)
    fleet_rows = pd.DataFrame()
    if csv_path is not None:
        fleet_rows = _fleet_drive_rows(csv_path, variant)

    if not all_rows.empty or not fleet_rows.empty:
        parts.append("<div class='card'>")
        parts.append("<h2>Time per attribute bucket — vehicle vs fleet</h2>")
        parts.append("<div style='color:#9ab;font-size:12px;margin-bottom:8px'>"
                     "% of drive time spent in each value range. All-trips "
                     f"({len(trips)}) and elevated ({n_elev}) columns are this "
                     "vehicle; fleet column is the rest of the variant. Same "
                     "denominator everywhere so distributions compare directly."
                     "</div>")
        for col, label, edges in BUCKET_DEFS:
            all_p = _pct_in_buckets(all_rows, col, edges)
            elev_p = _pct_in_buckets(elev_rows, col, edges)
            fleet_p = _pct_in_buckets(fleet_rows, col, edges)
            parts.append(f"<h3>{label}</h3>")
            parts.append("<table><tr><th>Bucket</th>"
                         f"<th>All trips ({len(trips)}) — % of time</th>"
                         f"<th>Elevated ({n_elev} trips) — % of time</th>"
                         f"<th>Fleet — % of drive time</th></tr>")
            for i in range(len(edges) - 1):
                parts.append(
                    f"<tr><td class='attr'>{_bucket_label(edges, i)}</td>"
                    f"<td>{all_p[i]:.1f}%</td>"
                    f"<td>{elev_p[i]:.1f}%</td>"
                    f"<td>{fleet_p[i]:.1f}%</td></tr>"
                )
            parts.append("</table>")
        parts.append("</div>")

    # ── 2D heatmaps: fleet | vehicle all trips | vehicle elevated trips ─
    if not all_rows.empty and not fleet_rows.empty:
        parts.append("<div class='card'>")
        parts.append("<h2>2D heatmaps — fleet vs vehicle "
                     "<span style='color:#667;font-size:11px;font-weight:normal'>"
                     "(select pair from dropdown)</span></h2>")
        parts.append("<div style='color:#9ab;font-size:12px;margin-bottom:8px'>"
                     "Each panel triplet shows the joint distribution of two "
                     "signals side-by-side: fleet drive rows on the left, this "
                     "vehicle's all-trips rows in the middle, its elevated-trips "
                     "rows on the right. Cells are % of rows so all panels are "
                     "directly comparable.</div>")
        # Build dropdown + image stack — only the selected one is visible.
        heatmap_items: list[tuple[str, str]] = []
        for x_col, y_col, x_lab, y_lab in HEATMAP_PAIRS:
            panels = [
                (fleet_rows, f"Fleet — {variant}"),
                (all_rows,   f"Vehicle all ({len(trips)} trips)"),
            ]
            if not elev_rows.empty:
                panels.append((elev_rows, f"Vehicle elevated ({n_elev} trips)"))
            png = _heatmap_png(panels, x_col, y_col, x_lab, y_lab)
            if png:
                heatmap_items.append((f"{x_lab} × {y_lab}", png))
        if heatmap_items:
            parts.append("<select id='hmSel' style='padding:4px 8px;background:#1f2532;"
                         "color:#e6e8ee;border:1px solid #2a2f3a;border-radius:4px;"
                         "font-size:13px;margin-bottom:10px'>")
            for i, (label, _) in enumerate(heatmap_items):
                parts.append(f"<option value='{i}'>{label}</option>")
            parts.append("</select>")
            for i, (label, png) in enumerate(heatmap_items):
                shown = "" if i == 0 else "display:none"
                parts.append(
                    f"<div class='hm-panel' data-idx='{i}' style='{shown}'>"
                    f"<img class='chart' src='data:image/png;base64,{png}' />"
                    f"</div>"
                )
        parts.append("</div>")

    # ── Per-trip deviation charts (dropdown) ────────────────────────────
    trip_charts: list[tuple[str, str]] = []  # (label, base64-png)
    for t, rows in zip(trips, per_trip_rows):
        if not rows:
            continue
        is_elev = _is_trip_elevated(t)
        tid = int(t.get("trip_id") or 0)
        title = (f"Trip {tid} — {_fmt(t.get('duration_min'))} min, "
                 f"{_fmt(t.get('dist_km'))} km"
                 + ("  [ELEV]" if is_elev else ""))
        png = _trip_chart_png(rows, title)
        if png:
            trip_charts.append((f"Trip {tid}" + (" [ELEV]" if is_elev else ""), png))
    if trip_charts:
        parts.append("<div class='card'>")
        parts.append("<h2>Per-trip deviation from fleet "
                     "<span style='color:#667;font-size:11px;font-weight:normal'>"
                     f"({len(trip_charts)} trips — pick one from dropdown)"
                     "</span></h2>")
        parts.append("<div style='color:#9ab;font-size:12px;margin-bottom:8px'>"
                     "Horizontal bar chart: each attribute's %-deviation from "
                     "the fleet median for that trip's (duration, SOC) bucket. "
                     "Red = alert (peers don't fire here), orange = warn, "
                     "green = in line with peers.</div>")
        parts.append("<select id='tripSel' style='padding:4px 8px;background:#1f2532;"
                     "color:#e6e8ee;border:1px solid #2a2f3a;border-radius:4px;"
                     "font-size:13px;margin-bottom:10px'>")
        for i, (label, _) in enumerate(trip_charts):
            parts.append(f"<option value='{i}'>{label}</option>")
        parts.append("</select>")
        for i, (label, png) in enumerate(trip_charts):
            shown = "" if i == 0 else "display:none"
            parts.append(
                f"<div class='trip-panel' data-idx='{i}' style='{shown}'>"
                f"<img class='chart' src='data:image/png;base64,{png}' />"
                f"</div>"
            )
        parts.append("</div>")

    parts.append(f"<div class='footer'>peer_check.py — fleet medians from "
                 f"{fleet_path.name}. A trip is 'elevated' if ≥1 danger "
                 "detector (D3/D4/D5/D8) fired on it.</div>")

    # ── Embed trip data + filter JS ────────────────────────────────────
    trips_js: list[dict[str, Any]] = []
    for t in trips:
        m0 = float(t.get("mode_0_frac") or 0.0)
        m1 = float(t.get("mode_1_frac") or 0.0)
        rec: dict[str, Any] = {
            "trip_id": int(t.get("trip_id") or 0),
            "mode": "thunder" if m1 >= m0 else "eco",
            "elev": 1 if _is_trip_elevated(t) else 0,
        }
        for a in DANGER_ATTRIBUTES:
            v = t.get(a)
            rec[a] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None
        trips_js.append(rec)
    parts.append("<script>")
    parts.append(f"const TRIPS={json.dumps(trips_js)};")
    parts.append(f"const ATTRS={json.dumps(list(DANGER_ATTRIBUTES))};")
    parts.append("""
function _fmt(v){ return (v==null||!isFinite(v))?'—':Number(v).toFixed(2); }
function applyFilters(){
  const m = document.querySelector('input[name=modeFlt]:checked').value;
  const v = document.querySelector('input[name=vrdFlt]:checked').value;
  document.querySelectorAll('tr.trip-row').forEach(r=>{
    const ok = (m==='all'||r.dataset.mode===m) && (v==='all'||r.dataset.elev===v);
    r.style.display = ok ? '' : 'none';
  });
  const filtered = TRIPS.filter(t =>
    (m==='all'||t.mode===m) && (v==='all'||String(t.elev)===v));
  document.getElementById('visTrips').textContent = filtered.length;
  ATTRS.forEach(attr=>{
    const vals = filtered.map(t=>t[attr]).filter(x=>x!=null&&isFinite(x));
    const avg = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
    const cell = document.querySelector("#attrAvgTable tr[data-attr='"+attr+"'] td.veh-avg");
    if(cell) cell.textContent = _fmt(avg);
  });
}
document.querySelectorAll('input[name=modeFlt], input[name=vrdFlt]').forEach(el=>
  el.addEventListener('change', applyFilters));
document.getElementById('totTrips').textContent = TRIPS.length;
applyFilters();

// Dropdown toggle for heatmaps + per-trip charts
function _panelToggle(selectId, panelClass){
  const sel = document.getElementById(selectId);
  if(!sel) return;
  sel.addEventListener('change', e=>{
    const idx = e.target.value;
    document.querySelectorAll('.'+panelClass).forEach(p=>{
      p.style.display = (p.dataset.idx === idx) ? '' : 'none';
    });
  });
}
_panelToggle('hmSel', 'hm-panel');
_panelToggle('tripSel', 'trip-panel');
""")
    parts.append("</script>")
    parts.append("</body></html>")
    return "".join(parts)


# ── Entry point ────────────────────────────────────────────────────────
def inspect(imei: str,
            csv_path: Path = C.INPUT_CSV,
            stage2_path: Path = C.STAGE2_OUT,
            fleet_path: Path | None = None,
            out_html: Path | None = None) -> Path:
    if fleet_path is None:
        fleet_path = C.STAGE2_OUT.parent / "fleet_reference.csv"
    if out_html is None:
        out_html = C.STAGE2_OUT.parent / f"peer_check_{imei}.html"

    fleet = _load_fleet_ref(fleet_path)
    variant, trips, raw_by_trip = _compute_trips_for_imei(imei, csv_path)
    if variant is None:
        raise SystemExit(f"[peer_check] IMEI {imei} not found in telemetry {csv_path}")
    if not trips:
        raise SystemExit(f"[peer_check] IMEI {imei} ({variant}) has no valid trips "
                         f"(min {C.TRIP_MIN_DURATION_MIN} min / {C.TRIP_MIN_DISTANCE_KM} km).")
    print(f"[peer_check] {imei}: variant={variant}, valid trips={len(trips)}")

    stage2 = _load_stage2_for_imei(imei, stage2_path)
    if stage2 is None:
        print(f"[peer_check] note: imei not found in {stage2_path.name} — verdict unknown")

    per_trip_rows: list[list[dict[str, Any]]] = []
    per_trip_detectors: list[list[dict[str, Any]]] = []
    for t in trips:
        bucket = _bucket_lookup(fleet, variant, t["duration_min"], t.get("soc_consumed", float("nan")))
        per_trip_rows.append(_compare_trip(t, bucket))
        per_trip_detectors.append(_compare_detector_events(t, bucket))

    html = render_html(imei, variant, stage2, trips,
                       per_trip_rows, per_trip_detectors, fleet_path,
                       fleet_df=fleet,
                       raw_by_trip=raw_by_trip,
                       stage2_path=stage2_path,
                       csv_path=csv_path)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    print(f"[peer_check] wrote {out_html}")
    return out_html


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Single-IMEI peer-comparison report.")
    p.add_argument("imei", help="vehicle IMEI to inspect")
    p.add_argument("--csv", type=Path, default=C.INPUT_CSV)
    p.add_argument("--stage2", type=Path, default=C.STAGE2_OUT)
    p.add_argument("--fleet", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="output HTML path")
    args = p.parse_args(argv)
    inspect(args.imei, csv_path=args.csv, stage2_path=args.stage2,
            fleet_path=args.fleet, out_html=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
