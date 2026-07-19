"""stage2_to_v3.py — bridge prod stage-2 output → v3 dashboard schema.

The legacy v3 pipeline (`Task3/validate_anomalies_v3.py`) produces
`anomaly_validation_v3.json`, which the dashboard exporter (`export_v3.py`)
merges into `dashboard_template_v3.html`. The prod pipeline
(`prod/stage2_validator.py`) writes the same per-vehicle facts into
`prod/out/stage2_verdicts.json` in a slightly different shape.

This module reads the prod JSON and writes the v3 JSON so the dashboard
can be rebuilt without re-running the slow per-vehicle CUSUM/IsolationForest
loop in `validate_anomalies_v3.py`.

Vehicle records and trip records are passed through unchanged except for
adding the `strong_metrics` label string the template renders. Summary keys
are remapped to the names `export_v3.py` reads (`high`, `no_valid_trips`,
`normal_after_validation`, `watchlist_total`, ...).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import config as C
from .explain import explain_vehicle


ROOT = C.STAGE2_OUT.parent.parent.parent  # Task3/
DEFAULT_PROD = C.STAGE2_OUT
DEFAULT_V3 = ROOT / "anomaly_validation_v3.json"


def _strong_metrics_label(strong_ids: list[str] | str) -> str:
    """Map ['d4_current_torque', 'd6_torque_speed'] → '<label1> | <label2>'."""
    if not strong_ids:
        return ""
    if isinstance(strong_ids, str):
        # Already a string — pass through.
        return strong_ids
    label_lookup = {d["id"]: d["label"] for d in C.DUAL_DETECTORS}
    return " | ".join(label_lookup.get(s, s) for s in strong_ids if s)


def _adapt_vehicle(v: dict[str, Any]) -> dict[str, Any]:
    out = dict(v)  # shallow copy preserves trips list reference (fine — read-only)
    if "strong_metrics" not in out:
        out["strong_metrics"] = _strong_metrics_label(out.get("strong_metric_ids", []))
    out.setdefault("review_recommendation", "")
    out.setdefault("timeline", None)
    # Plain-English explanation bullets, used by the dashboard validation card
    # and by peer_check reports. Multi-line bullets keep "\n" so the renderer
    # can choose how to wrap them.
    out["explanations"] = explain_vehicle(v)
    return out


def _adapt_summary(prod_summary: dict[str, Any], vehicles: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild summary in the shape `export_v3.py` reads."""
    elevated = sum(1 for v in vehicles if v.get("verdict") == "ELEVATED")
    watch = sum(1 for v in vehicles if v.get("verdict") == "WATCH")
    normal = sum(1 for v in vehicles if v.get("verdict") == "NORMAL")
    no_valid_trips = sum(1 for v in vehicles if v.get("watch_reason") == "DATA_GAP_NO_VALID_TRIPS"
                                                  or v.get("watch_reason") == "NO_VALID_TRIPS")
    detector_flagged = sum(1 for v in vehicles if v.get("detector_flag"))
    spirited_total = sum(1 for v in vehicles if v.get("spirited"))
    spirited_demoted = sum(1 for v in vehicles
                            if v.get("spirited") and v.get("verdict") == "NORMAL")
    valid_trip_count = sum(int(v.get("n_trips") or 0) for v in vehicles)
    vehicles_with_valid_trips = sum(1 for v in vehicles if (v.get("n_trips") or 0) > 0)

    summary: dict[str, Any] = {
        "pipeline_version": prod_summary.get("pipeline_version", "v5-prod") + "-via-adapter",
        "total_validated": len(vehicles),
        "dashboard_vehicles": len(vehicles),                # exporter merges 1-to-1
        "dashboard_joined_to_validator": len(vehicles),
        "detector_flagged": detector_flagged,
        # Verdict-tier counts the exporter reads:
        "high": 0,                                         # v5 prod has no HIGH tier
        "elevated": elevated,
        "watch": watch,
        "watchlist_total": elevated + watch,
        "no_valid_trips": no_valid_trips,
        "normal_after_validation": normal,
        # Spirited-driver guard reporting:
        "spirited_total": spirited_total,
        "spirited_demoted_to_normal": spirited_demoted,
        # Trip rollup:
        "valid_trips": valid_trip_count,
        "vehicles_with_valid_trips": vehicles_with_valid_trips,
        # Pass-through aggregates the prod summary already computed:
        "verdict_counts": prod_summary.get("verdict_counts", {}),
        "watch_reason_counts": prod_summary.get("watch_reason_counts", {}),
        "variant_verdict_counts": prod_summary.get("variant_verdict_counts", {}),
        "params": prod_summary.get("params", {}),
    }
    return summary


def adapt(prod_path: Path = DEFAULT_PROD,
          v3_path: Path = DEFAULT_V3) -> Path:
    print(f"[stage2_to_v3] reading {prod_path}")
    prod = json.loads(prod_path.read_text())
    vehicles_in = prod.get("vehicles", [])
    vehicles_out = [_adapt_vehicle(v) for v in vehicles_in]
    summary_out = _adapt_summary(prod.get("summary", {}), vehicles_out)

    v3_payload = {
        "summary": summary_out,
        "vehicles": vehicles_out,
    }
    v3_path.parent.mkdir(parents=True, exist_ok=True)
    v3_path.write_text(json.dumps(v3_payload, separators=(",", ":")))
    print(f"[stage2_to_v3] wrote {v3_path} — "
          f"{len(vehicles_out)} vehicles "
          f"(elevated={summary_out['elevated']}, watch={summary_out['watch']}, "
          f"normal={summary_out['normal_after_validation']}, "
          f"no_valid_trips={summary_out['no_valid_trips']})")
    return v3_path


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prod", type=Path, default=DEFAULT_PROD,
                   help="prod stage-2 verdicts JSON (default: prod/out/stage2_verdicts.json)")
    p.add_argument("--out", type=Path, default=DEFAULT_V3,
                   help="path for legacy v3 JSON (default: anomaly_validation_v3.json)")
    args = p.parse_args(argv)
    adapt(args.prod, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
