"""v3_export.py — translate v5 prod verdicts into v3-dashboard-compatible JSON.

Reads `prod/out/stage2_verdicts.json` (v5 schema with `spirited` field, no
NO_VALID_TRIPS rows) and writes `Task3/anomaly_validation_v3.json` so the
existing `export_v3.py` + `dashboard_template_v3.html` pair render with the
v5 verdicts. The v3 dashboard already maps {ELEVATED, WATCH, NORMAL} buckets
correctly — we just need a summary block and per-vehicle rows in v3 shape.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import config as C


def _load(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_v3_payload(stage2: dict) -> dict[str, Any]:
    vehicles = stage2.get("vehicles", [])
    summary_in = stage2.get("summary", {})

    # Per-vehicle: keep the v5 schema (it's already a superset of v3 fields),
    # add `severity` (used by some template branches) and `validation_run` flag.
    out_vehicles = []
    for v in vehicles:
        rec = dict(v)  # shallow copy
        verdict = rec.get("verdict", "NORMAL")
        # severity is what the detector-side cards use; map verdict→severity:
        # ELEVATED→HIGH, WATCH→MEDIUM, NORMAL→NORMAL.
        rec["severity"] = {"ELEVATED": "HIGH", "WATCH": "MEDIUM",
                           "NORMAL": "NORMAL"}.get(verdict, "NORMAL")
        # `strong_metrics` (human-readable) is what the template displays;
        # build it from `strong_metric_ids` if missing.
        if "strong_metrics" not in rec or not rec["strong_metrics"]:
            ids = rec.get("strong_metric_ids") or []
            labels = [C.METRIC_LABELS.get(m, m) for m in ids]
            rec["strong_metrics"] = "; ".join(labels) if labels else ""
        out_vehicles.append(rec)

    verdict_counts = Counter(r.get("verdict", "NORMAL") for r in out_vehicles)
    watch_reason_counts = Counter(r.get("watch_reason") for r in out_vehicles
                                   if r.get("verdict") == "WATCH")
    by_variant: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in out_vehicles:
        by_variant[r.get("variant", "")][r.get("verdict", "")] += 1

    spirited_total = sum(1 for r in out_vehicles if r.get("spirited"))
    spirited_normal = sum(1 for r in out_vehicles
                          if r.get("spirited") and r.get("verdict") == "NORMAL")

    summary_out = {
        "total_validated": len(out_vehicles),
        "dashboard_vehicles": len(out_vehicles),
        "dashboard_joined_to_validator": len(out_vehicles),
        "detector_flagged": sum(1 for r in out_vehicles if r.get("detector_flag")),
        "high": 0,  # v5 has no HIGH tier — collapsed into ELEVATED
        "elevated": int(verdict_counts.get("ELEVATED", 0)),
        "watch": int(verdict_counts.get("WATCH", 0)),
        "watchlist_total": int(verdict_counts.get("ELEVATED", 0) + verdict_counts.get("WATCH", 0)),
        "no_valid_trips": 0,  # filtered out in v5
        "normal_after_validation": int(verdict_counts.get("NORMAL", 0)),
        "spirited_total": spirited_total,
        "spirited_demoted_to_normal": spirited_normal,
        "verdict_counts": {str(k): int(v) for k, v in verdict_counts.items()},
        "watch_reason_counts": {str(k): int(v) for k, v in watch_reason_counts.items()},
        "variant_verdict_counts": {k: dict(v) for k, v in by_variant.items()},
        "valid_trips": int(summary_in.get("valid_trips", 0)) if summary_in else 0,
        "vehicles_with_valid_trips": len(out_vehicles),
        "params": dict(summary_in.get("params", {})) if summary_in else {},
        "pipeline_version": C.PIPELINE_VERSION,
    }

    return {"summary": summary_out, "vehicles": out_vehicles}


def run(stage2_path: Path | None = None,
        out_path: Path | None = None) -> dict[str, Any]:
    if stage2_path is None:
        stage2_path = C.STAGE2_OUT
    if out_path is None:
        out_path = C.ROOT / "anomaly_validation_v3.json"

    print(f"[V3-EXPORT] reading {stage2_path}")
    stage2 = _load(stage2_path)
    payload = build_v3_payload(stage2)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    s = payload["summary"]
    print(f"[V3-EXPORT] wrote {out_path}")
    print(f"[V3-EXPORT] vehicles={s['total_validated']:,} | "
          f"ELEVATED={s['elevated']} | WATCH={s['watch']} | "
          f"NORMAL={s['normal_after_validation']} | "
          f"spirited_demoted={s['spirited_demoted_to_normal']}")
    return payload


if __name__ == "__main__":
    run()
