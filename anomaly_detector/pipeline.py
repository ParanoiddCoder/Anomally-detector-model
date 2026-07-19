"""
pipeline.py — end-to-end runner: Stage-1 ∥ Stage-2 prep → merge → JSON.

Optimisation: the telemetry CSV is read once (by stage2_validator.load_telemetry,
which already does chunked float-downcast loading). Stage-1 feature extraction
and Stage-2 telemetry prep (variant stats, trip table, charge events) are then
run in parallel threads so CPU-bound work overlaps.  validate() merges the two
results and the rest of the pipeline is unchanged.

Usage:
    python -m anomaly_detector.pipeline                # run everything
    python -m anomaly_detector.pipeline --dashboard    # also build the HTML preview
    python -m anomaly_detector.pipeline --stage 2      # Stage-2 only (assumes stage1_flags.json exists)
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import config as C
from . import stage1_detector, stage2_validator


# ── Stage-1 helpers that accept an in-memory DataFrame ────────────────────────

def _run_stage1_from_df(telemetry: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Run Stage-1 on an already-loaded telemetry DataFrame and write the JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    print(f"[STAGE-1] rows={len(telemetry):,} vehicles={telemetry['imei'].nunique():,}")
    all_results: list[pd.DataFrame] = []
    for variant, vdf in telemetry.groupby("variant", sort=False):
        feats = stage1_detector._per_vehicle_features(vdf)
        if feats.empty:
            continue
        scored = stage1_detector._score_variant(feats)
        all_results.append(scored)
        n_anom = int(scored["is_anomaly"].sum())
        k_used = int(scored["cluster_id"].nunique())
        print(f"  [S1] {variant}: vehicles={len(scored):,} anomalies={n_anom:,} k={k_used}")

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


def _run_stage2_prep(telemetry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute variant stats, trip table, and charge events from telemetry."""
    variant_stats = stage2_validator.compute_variant_stats(telemetry)
    trips = stage2_validator.build_trip_table(telemetry, variant_stats)
    charges = stage2_validator.build_charge_events(telemetry, variant_stats)
    n_vehicles = trips["imei"].nunique() if len(trips) else 0
    print(f"  [S2] valid trips={len(trips):,} vehicles_with_trips={n_vehicles:,}")
    return trips, charges, variant_stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(C.INPUT_CSV))
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    p.add_argument("--dashboard", action="store_true", help="Build + open the HTML preview")
    p.add_argument("--no-open", action="store_true", help="With --dashboard, do not open the browser")
    args = p.parse_args()

    csv_path = Path(args.csv)
    t0 = time.time()

    if args.stage == "1":
        # Stage-1 only: use its own loader (keeps this path self-contained)
        stage1_detector.run(csv_path=csv_path, out_path=C.STAGE1_OUT)

    elif args.stage == "2":
        # Stage-2 only: assumes stage1_flags.json already exists
        stage2_validator.run(
            csv_path=csv_path,
            stage1_path=C.STAGE1_OUT,
            out_json=C.STAGE2_OUT,
            out_summary=C.SUMMARY_OUT,
        )

    else:
        # ── Optimised "all" path ──────────────────────────────────────────
        # 1. Load telemetry once (Stage-2 loader has chunked + typed reading)
        print(f"[PIPELINE] loading {csv_path.name}")
        telemetry = stage2_validator.load_telemetry(csv_path)
        print(f"[PIPELINE] loaded {len(telemetry):,} rows, {telemetry['imei'].nunique():,} vehicles")

        # 2. Stage-1 feature extraction and Stage-2 telemetry prep run in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_s1 = pool.submit(_run_stage1_from_df, telemetry, C.STAGE1_OUT)
            fut_s2 = pool.submit(_run_stage2_prep, telemetry)

            # Collect results as they finish; log any error immediately
            stage1_flags: pd.DataFrame | None = None
            trips = charges = None
            for fut in as_completed([fut_s1, fut_s2]):
                if fut is fut_s1:
                    stage1_flags = fut.result()   # raises if Stage-1 crashed
                else:
                    trips, charges, _ = fut.result()

        # 3. Build profiles and run the verdict gate (sequential — depends on both)
        stage1_df = stage2_validator.load_stage1(C.STAGE1_OUT)
        profiles = stage2_validator.build_profiles(trips, charges)
        results = stage2_validator.validate(stage1_df, profiles, trips=trips)
        summary = stage2_validator.build_summary(results)
        stage2_validator.export(results, summary, C.STAGE2_OUT, C.SUMMARY_OUT)

        vc = results["verdict"].value_counts().to_dict()
        print("[VERDICT] " + ", ".join(f"{k}={v}" for k, v in sorted(vc.items(), key=lambda kv: -kv[1])))

    if args.dashboard and args.stage in ("2", "all"):
        from . import dashboard_export
        dashboard_export.generate(open_browser=not args.no_open)

    print(f"[PIPELINE] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
