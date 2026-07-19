"""fleet_reference.py — per-variant fleet behavior table.

For each (variant, duration_bin, soc_consumed_bin) bucket, computes the median of
trip-level performance attributes across the fleet. Output is a single CSV
that engineers can hold next to any vehicle's trips and ask:

    "On a 30-min / 6%-SOC DemoEV|Std trip, what does the fleet typically do?"

SOC-consumed binning groups trips by how much energy was actually used,
which is a more meaningful peer comparison than raw distance (two trips of
the same distance can consume very different energy depending on mode and
motor health).

Bins:
  duration_bin     = round(duration_min   / FLEET_DURATION_BIN_MIN) * FLEET_DURATION_BIN_MIN
  soc_consumed_bin = round(soc_consumed   / FLEET_SOC_BIN_PCT)      * FLEET_SOC_BIN_PCT

Buckets with fleet_n < FLEET_MIN_BUCKET_N are dropped.
Buckets with fleet_n < FLEET_LOW_CONFIDENCE_N are flagged low_confidence=True.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C
from .stage2_validator import (
    build_trip_table,
    compute_variant_stats,
    load_telemetry,
)

# Trip-level attributes whose medians the fleet table reports.
# Names match keys produced by stage2_validator._trip_metrics().
FLEET_ATTRIBUTES = [
    "mean_speed",
    "p90_speed",
    "median_rpm",
    "p90_rpm",
    "median_torque",
    "p90_torque",
    "median_throttle",
    "p90_throttle",
    "low_speed_frac",
    "median_current",
    "p90_current",
    "median_temp",
    "peak_temp",
    "mode_0_frac",   # fraction of trip rows in Eco mode
    "mode_1_frac",   # fraction of trip rows in Thunder mode
]

# Per-trip dual-detector firing counts. Same trip-level granularity as
# FLEET_ATTRIBUTES but they're integer event counts, so peer_check renders
# them in a separate "Detector firings" section with its own status logic.
DETECTOR_EVENT_ATTRS = [
    "d1_rpm_speed_events",
    "d2_throttle_speed_events",
    "d3_throttle_current_events",
    "d4_current_torque_events",
    "d5_current_rpm_torque_events",
    "d6_torque_speed_events",
    "d8_temp_overheat_events",
]


def _bin(series: pd.Series, width: int) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") / width).round().astype("Int64") * width


def build_fleet_reference(trips: pd.DataFrame) -> pd.DataFrame:
    """Per-variant, per-(duration_bin, soc_consumed_bin) fleet medians.

    `trips` is the trip table produced by stage2_validator.build_trip_table.
    Peers are grouped by energy consumed (SOC %) rather than distance so that
    trips with similar electrical demand are compared against each other.
    """
    if trips.empty:
        return pd.DataFrame()

    df = trips.copy()
    df["duration_bin"] = _bin(df["duration_min"], C.FLEET_DURATION_BIN_MIN)
    df["soc_consumed_bin"] = _bin(df["soc_consumed"], C.FLEET_SOC_BIN_PCT)
    df = df.dropna(subset=["duration_bin", "soc_consumed_bin"])

    attrs = [a for a in (FLEET_ATTRIBUTES + DETECTOR_EVENT_ATTRS) if a in df.columns]

    grouped = df.groupby(["variant", "duration_bin", "soc_consumed_bin"], dropna=False)
    agg = grouped[attrs].median().reset_index()
    agg["fleet_n"] = grouped.size().values

    agg = agg[agg["fleet_n"] >= C.FLEET_MIN_BUCKET_N].copy()
    agg["low_confidence"] = agg["fleet_n"] < C.FLEET_LOW_CONFIDENCE_N

    rename = {a: f"median_{a}" for a in attrs}
    agg = agg.rename(columns=rename)

    cols = ["variant", "duration_bin", "soc_consumed_bin", "fleet_n", "low_confidence"] \
           + [f"median_{a}" for a in attrs]
    return agg[cols].sort_values(["variant", "duration_bin", "soc_consumed_bin"]).reset_index(drop=True)


def run(csv_path: Path = C.INPUT_CSV,
        out_csv: Path | None = None) -> pd.DataFrame:
    if out_csv is None:
        out_csv = C.STAGE2_OUT.parent / "fleet_reference.csv"
    print(f"[FLEET-REF] loading {csv_path.name}")
    telemetry = load_telemetry(csv_path)
    variant_stats = compute_variant_stats(telemetry)
    trips = build_trip_table(telemetry, variant_stats)
    print(f"[FLEET-REF] trips loaded: {len(trips):,} across "
          f"{trips['variant'].nunique()} variants")

    ref = build_fleet_reference(trips)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(out_csv, index=False)
    print(f"[FLEET-REF] wrote {out_csv} — {len(ref)} buckets, "
          f"{int(ref['low_confidence'].sum())} flagged low_confidence")
    return ref


if __name__ == "__main__":
    run()
