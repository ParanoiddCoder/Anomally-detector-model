"""validate_elevated.py — generate peer_check reports for every ELEVATED vehicle.

Reads the current `prod/out/stage2_verdicts.json`, picks every vehicle whose
verdict is ELEVATED, and runs `prod.peer_check.inspect` on each. Reports are
written to `prod/out/peer_check_<imei>.html` and a small `peer_check_index.html`
is produced that links to all of them.

    python -m anomaly_detector.validate_elevated
    python -m anomaly_detector.validate_elevated --verdict WATCH        # validate WATCH instead
    python -m anomaly_detector.validate_elevated --variant 'DemoEV | Std' # restrict to a variant
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as C
from .peer_check import inspect


def _load_targets(verdict: str, variant: str | None) -> list[dict]:
    payload = json.loads(C.STAGE2_OUT.read_text())
    out: list[dict] = []
    for v in payload.get("vehicles", []):
        if v.get("verdict") != verdict:
            continue
        if variant and v.get("variant") != variant:
            continue
        out.append(v)
    out.sort(key=lambda v: -int(v.get("strong_metric_count") or 0))
    return out


def _index_html(verdict: str, rows: list[dict], out_dir: Path) -> Path:
    path = out_dir / "peer_check_index.html"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{verdict} validation — {len(rows)} vehicles</title>",
        "<style>",
        "body{background:#0e1014;color:#e6e8ee;font-family:-apple-system,system-ui,sans-serif;padding:24px}",
        "h1{font-size:20px;margin:0 0 14px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "th,td{border-bottom:1px solid #2a2f3a;padding:6px 10px;text-align:left}",
        "th{background:#1f2532;color:#9ab}",
        "tr:hover td{background:rgba(255,255,255,.03)}",
        "a{color:#7ab7ff;text-decoration:none}",
        "a:hover{text-decoration:underline}",
        "</style></head><body>",
        f"<h1>{verdict} validation — {len(rows)} vehicle(s)</h1>",
        "<table><tr><th>#</th><th>IMEI</th><th>Variant</th>"
        "<th>n_trips</th><th>n_dual</th><th>Reason</th>"
        "<th>Strong detectors</th><th>Report</th></tr>",
    ]
    for i, v in enumerate(rows, 1):
        imei = v["imei"]
        rel = f"peer_check_{imei}.html"
        strong = ",".join(v.get("strong_metric_ids") or []) or "—"
        parts.append(
            f"<tr><td>{i}</td>"
            f"<td>{imei}</td>"
            f"<td>{v.get('variant', '—')}</td>"
            f"<td>{v.get('n_trips', 0)}</td>"
            f"<td>{v.get('strong_metric_count', 0)}</td>"
            f"<td>{v.get('watch_reason', '—')}</td>"
            f"<td>{strong}</td>"
            f"<td><a href='{rel}'>open</a></td></tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("".join(parts))
    return path


def run(verdict: str = "ELEVATED", variant: str | None = None) -> Path:
    rows = _load_targets(verdict, variant)
    if not rows:
        print(f"[validate] no vehicles with verdict={verdict}"
              + (f", variant={variant}" if variant else ""))
        return C.STAGE2_OUT.parent / "peer_check_index.html"

    out_dir = C.STAGE2_OUT.parent
    print(f"[validate] running peer_check on {len(rows)} {verdict} vehicle(s)...")
    for v in rows:
        try:
            inspect(v["imei"])
        except SystemExit as e:
            print(f"  ! {v['imei']} skipped: {e}")
        except Exception as e:
            print(f"  ! {v['imei']} failed: {type(e).__name__}: {e}")
    index_path = _index_html(verdict, rows, out_dir)
    print(f"[validate] wrote index: {index_path}")
    print(f"[validate] open it with:  open {index_path}")
    return index_path


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--verdict", default="ELEVATED",
                   choices=["ELEVATED", "WATCH", "NORMAL"])
    p.add_argument("--variant", default=None,
                   help="restrict to one variant (e.g. 'DemoEV | Std')")
    args = p.parse_args(argv)
    run(args.verdict, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
