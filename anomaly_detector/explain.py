"""explain.py — plain-English explanations of why a vehicle was tagged.

Reads a single vehicle record from `prod/out/stage2_verdicts.json` (or the
adapter output `anomaly_validation_v3.json`) and produces a list of bullet
points that say, in normal English:

  • what verdict the vehicle got and why,
  • what each strong dual-signal detector means physically,
  • how strongly the evidence deviates from the peer fleet,
  • when the spirited-driver guard fired or was overridden.

The explanations are written to be readable by someone who hasn't seen the
detector code — no `d4_current_torque_z = 2.7` jargon, just sentences.
"""
from __future__ import annotations

from typing import Any


# Per-detector physical meaning.
# `headline` = one-line description of the failure mode the detector targets.
# `evidence` = template that turns the per-vehicle numbers into English.
DETECTOR_META: dict[str, dict[str, str]] = {
    "d1_rpm_speed": {
        "headline": "Motor RPM stays high while wheel speed stays low",
        "meaning": "classic drivetrain-slip signature (clutch, belt or coupling not transferring power to the wheels). Persistent dwells across multiple trips usually mean drivetrain wear; brief single-trip bursts can be hard-launch artefacts.",
    },
    "d2_throttle_speed": {
        "headline": "Throttle pressed hard but the vehicle barely accelerates",
        "meaning": "driver demand is there, but the powertrain isn't producing motion. Could be controller cut-out under load, a stuck contactor, or thermal de-rating.",
    },
    "d3_throttle_current": {
        "headline": "Throttle high but battery current stays low",
        "meaning": "the controller is being asked for power but isn't drawing it from the pack — a controller fault, fuse / contactor issue, or current-sensor problem.",
    },
    "d4_current_torque": {
        "headline": "Battery current is high but motor torque is low",
        "meaning": "the motor is consuming energy without producing useful work — typical of motor winding degradation, demagnetised rotor, or shorted phase. The energy goes to heat instead of motion.",
    },
    "d5_current_rpm_torque": {
        "headline": "Current and RPM both spike but torque doesn't",
        "meaning": "freewheeling pattern — the motor is spinning fast and drawing current but not loaded. Common with broken driveline (broken shaft, slipping coupling) or sensor drift.",
    },
    "d6_torque_speed": {
        "headline": "High torque demanded but speed barely increases",
        "meaning": "powertrain is fighting drag — could be brake-binding, dragging caliper, or a mechanical resistance in the drivetrain. Sustained dwell is a strong wear signal.",
    },
    "d7_charge_overtaper": {
        "headline": "At >95% SoC the BMS is still pulling high charge current",
        "meaning": "the charger isn't tapering as the pack approaches full — BMS / cell-balancing fault or charger profile mismatch. Can accelerate cell aging.",
    },
    "d8_temp_overheat": {
        "headline": "Controller is running hot while drawing low current",
        "meaning": "heat without proportional load — pointing to a cooling-system fault (fan, coolant flow, thermal-paste bond) rather than overload. If left untreated, controller de-rating or shutdown follows.",
    },
}

WATCH_REASON_META: dict[str, str] = {
    "ML_PLUS_ONE_DUAL": "ML detector flagged this vehicle and exactly one dual-signal physics test fired — not enough corroborating physics evidence to ELEVATE under the v5 gate (which requires ≥2 strong duals).",
    "ML_ONLY_PATTERN": "ML detector flagged this vehicle but no dual-signal physics test fired strongly — the usage pattern looks unusual but no specific drivetrain / controller / thermal failure mode is confirmed.",
    "PHYSICS_ONLY_NO_ML": "One or more dual-signal physics tests fired, but the ML detector didn't flag the overall usage pattern — possibly a localized issue without broader drift in the vehicle's profile.",
    "INSUFFICIENT_PEERS": "Too few usage-matched peers in the same variant to do reliable peer-z scoring — verdict is conservative.",
    "NO_ML_NO_DUAL": "Neither the ML detector nor any physics test fired — vehicle behaves like a typical fleet member.",
    "ML_AND_DUAL": "ML detector flagged AND ≥2 dual-signal physics tests fired in the same vehicle — the v5 ELEVATED gate.",
    "SPIRITED_DRIVER_HIGH_SPEED_HIGH_RPM": "Driving pattern is dominated by fast cruising (≥60% of trips above 40 km/h mean and median p90 speed ≥55 km/h) and physics evidence is weak — demoted to NORMAL.",
}


def _strength(z: float | None) -> str:
    """Bucket a peer-z score into a plain-English strength."""
    if z is None or not isinstance(z, (int, float)):
        return "above the peer-fleet baseline"
    if z >= 4.0:
        return "extreme — far above any peer in the same variant"
    if z >= 3.0:
        return "very strong — top-1% in the peer fleet"
    if z >= 2.0:
        return "strong — clearly above the peer fleet"
    return "above the peer-fleet baseline"


def _detector_bullet(det_id: str, v: dict[str, Any]) -> str:
    meta = DETECTOR_META.get(det_id)
    rate = v.get(f"{det_id}_rate")
    z = v.get(f"{det_id}_z")
    events = v.get(f"{det_id}_events_total")
    max_dwell = v.get(f"{det_id}_max_dwell_s")

    if meta is None:
        return f"• {det_id} fired strongly (peer-z = {z:.1f}σ)." if z is not None else f"• {det_id} fired strongly."

    parts = [f"• {meta['headline']} — {meta['meaning']}"]
    evidence: list[str] = []
    if events is not None and events > 0:
        evidence.append(f"{int(events)} event{'s' if events != 1 else ''}")
    if max_dwell is not None and max_dwell > 0:
        evidence.append(f"longest stretch {max_dwell:.0f}s")
    if z is not None:
        evidence.append(f"{_strength(z)} (peer-z = {float(z):.1f}σ)")
    elif rate is not None and rate > 0:
        evidence.append(f"{float(rate):.2f} events/hr")
    if evidence:
        parts.append(f"  Evidence: {'; '.join(evidence)}.")
    return "\n".join(parts)


def explain_vehicle(v: dict[str, Any]) -> list[str]:
    """Return a list of plain-English bullets describing this vehicle's verdict."""
    verdict = v.get("verdict", "")
    reason = v.get("watch_reason", "") or v.get("reason", "")
    n_dual = int(v.get("strong_metric_count") or 0)
    spirited = bool(v.get("spirited"))
    n_trips = int(v.get("n_trips") or 0)
    n_peers = int(v.get("n_peers") or 0)
    detector_flag = bool(v.get("detector_flag"))
    strong_ids = v.get("strong_metric_ids") or []

    bullets: list[str] = []

    # ── Headline ─────────────────────────────────────────────────────
    if verdict == "ELEVATED":
        ml_part = "the ML usage-pattern detector flagged this vehicle" if detector_flag else "the ML detector did not flag it"
        bullets.append(
            f"Flagged ELEVATED. Both {ml_part} AND {n_dual} strong "
            f"dual-signal physics tests fired in the same vehicle. "
            f"Under the v5 gate, that combination is the highest-confidence anomaly tier."
        )
    elif verdict == "WATCH":
        meta_line = WATCH_REASON_META.get(reason, "Partial evidence — flagged for review.")
        bullets.append(f"Flagged WATCH ({reason}). {meta_line}")
    elif verdict == "NORMAL" and spirited:
        bullets.append(
            f"Verdict NORMAL (spirited-driver demote). At least 60% of this vehicle's trips "
            f"have mean speed above 40 km/h and the median trip p90 speed is ≥55 km/h, "
            f"which looks like a fast-driving pattern. Only {n_dual} strong dual signal "
            f"fired, which is below the carve-out threshold (≥2 duals would override "
            f"the demote). Conclusion: drives fast, but no confirmed mechanical evidence."
        )
    elif verdict == "NORMAL":
        bullets.append(
            "Verdict NORMAL. Neither the ML usage-pattern detector nor any "
            "dual-signal physics test fired strongly — this vehicle behaves like the rest of its variant peers."
        )
    else:
        bullets.append(f"Verdict {verdict or '—'} ({reason}).")

    # ── Per-detector physical narrative ──────────────────────────────
    for det_id in strong_ids:
        # `strong_metric_ids` come in as eg "d4_current_torque_rate" or "d4_current_torque" —
        # normalise by stripping any trailing "_rate" so it matches DETECTOR_META.
        key = det_id[:-5] if det_id.endswith("_rate") else det_id
        bullets.append(_detector_bullet(key, v))

    # ── Spirited carve-out call-out for ELEVATED ─────────────────────
    if spirited and verdict == "ELEVATED":
        bullets.append(
            "Note on driving style: this vehicle ALSO matches the spirited-driver "
            "pattern (≥60% trips above 40 km/h mean, median p90 ≥55 km/h). The v5 guard "
            "deliberately does NOT demote vehicles with ≥2 strong dual signals — the "
            "physics evidence overrides driver style, because fast driving can mask real "
            "powertrain faults rather than create them."
        )

    # ── Context ──────────────────────────────────────────────────────
    if n_peers and n_peers < 8 and verdict != "NORMAL":
        bullets.append(
            f"Caveat: only {n_peers} usage-matched peers were available in this variant, "
            f"so peer-z bounds are wider than usual — treat the magnitudes as indicative."
        )
    if n_trips <= 2 and verdict in {"ELEVATED", "WATCH"}:
        bullets.append(
            f"Caveat: this verdict is based on only {n_trips} valid discharge trip"
            f"{'s' if n_trips != 1 else ''}. More trips would tighten the peer comparison."
        )

    return bullets
