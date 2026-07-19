"""
dashboard_export.py — inject stage2_verdicts.json into the production HTML
template and produce a single self-contained file you can open locally.

Use this to eyeball the verdicts in a browser.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from . import config as C

ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT / "dashboard_template.html"
DEFAULT_OUTPUT = ROOT / "out" / "dashboard.html"


def generate(verdicts_path: Path = C.STAGE2_OUT,
             template_path: Path = DEFAULT_TEMPLATE,
             output_path: Path = DEFAULT_OUTPUT,
             open_browser: bool = True) -> None:
    if not verdicts_path.exists():
        raise FileNotFoundError(f"{verdicts_path} not found — run stage-2 first.")
    if not template_path.exists():
        raise FileNotFoundError(f"{template_path} not found.")

    payload = json.loads(verdicts_path.read_text())
    template = template_path.read_text()
    html = template.replace("{{DATA_JSON}}", json.dumps(payload, separators=(",", ":")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    size_kb = output_path.stat().st_size / 1024
    print(f"[DASHBOARD] {output_path} ({size_kb:.1f} KB)")
    if open_browser:
        webbrowser.open(f"file://{output_path.resolve()}")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verdicts", default=str(C.STAGE2_OUT))
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()
    generate(Path(args.verdicts), Path(args.template), Path(args.output),
             open_browser=not args.no_open)


if __name__ == "__main__":
    cli()
