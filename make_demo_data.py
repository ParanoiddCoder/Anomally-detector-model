"""
make_demo_data.py — generate a small synthetic telemetry CSV so the anomaly
detector runs end-to-end with no external data source.

The schema matches what `anomaly_detector.stage2_validator.load_telemetry`
expects. A handful of "vehicles" (imeis) drive several trips each; a few are
seeded with fault signatures (e.g. RPM high while speed stays low) so the
Stage-2 dual-signal detectors have something to catch.

Usage:
    python make_demo_data.py                 # writes data/demo_telemetry.csv
    python make_demo_data.py --vehicles 40   # more vehicles
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEC_PER_ROW = 5
STATE_IDLE, STATE_CHARGE, STATE_RUN = 0, 1, 2


def _trip(rng: np.random.Generator, start, faulty: bool) -> pd.DataFrame:
    """One driving trip: ~10-25 min of moving rows plus a short idle tail."""
    n = int(rng.integers(150, 320))          # 150*5s ≈ 12 min .. 320*5s ≈ 27 min
    t = pd.date_range(start, periods=n, freq=f"{SEC_PER_ROW}s", tz="UTC")

    speed = np.clip(rng.normal(30, 10, n), 0, 70).astype("float32")
    rpm = (speed * 90 + rng.normal(0, 200, n)).clip(0, 8000).astype("float32")
    throttle = np.clip(speed / 70 * 100 + rng.normal(0, 8, n), 0, 100).astype("float32")
    torque = np.clip(throttle * 1.5 + rng.normal(0, 10, n), 0, 200).astype("float32")
    current = np.clip(torque * 1.2 + rng.normal(0, 8, n), 0, 300).astype("float32")
    ctrl_temp = np.clip(rng.normal(45, 6, n), 20, 95).astype("float32")

    if faulty:
        # drivetrain-slip signature: RPM stays high while speed collapses
        hi = slice(n // 3, 2 * n // 3)
        speed[hi] = np.clip(speed[hi] * 0.2, 0, 70)
        rpm[hi] = np.clip(rpm[hi] * 1.3 + 1500, 0, 9000)
        torque[hi] = np.clip(torque[hi] * 1.4, 0, 220)

    dist_km = np.cumsum(speed) * (SEC_PER_ROW / 3600.0)
    odometer = (1000 + dist_km).astype("float32")
    soc = np.clip(90 - dist_km * 0.4 + rng.normal(0, 0.3, n), 5, 100).astype("float32")

    return pd.DataFrame({
        "time": t,
        "odometer": odometer,
        "rpm": rpm,
        "torque": torque,
        "controllerTemperature": ctrl_temp,
        "soc": soc,
        "speed": speed,
        "throttle": throttle,
        "current": current,
        "vehicleState": STATE_RUN,
        "controllerControllerMode": 1,
    })


def build(n_vehicles: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for v in range(n_vehicles):
        imei = f"{860000000000000 + v}"
        variant = rng.choice(["Std", "Long"])
        faulty_vehicle = v % 7 == 0                      # ~1 in 7 has faulty trips
        start = pd.Timestamp("2026-01-10 06:00:00", tz="UTC")
        for trip_i in range(int(rng.integers(3, 7))):
            df = _trip(rng, start, faulty=faulty_vehicle and trip_i % 2 == 0)
            df["imei"] = imei
            df["vehicle_model_name"] = "DemoEV"
            df["vehicle_variant_name"] = variant
            frames.append(df)
            start = df["time"].iloc[-1] + pd.Timedelta(hours=int(rng.integers(2, 8)))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "demo_telemetry.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build(args.vehicles, args.seed)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df):,} rows, {df['imei'].nunique()} vehicles)")


if __name__ == "__main__":
    main()
