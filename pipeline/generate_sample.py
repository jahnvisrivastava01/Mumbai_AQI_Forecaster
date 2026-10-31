"""Generate synthetic data so the project runs without an API key.

  python -m pipeline.generate_sample --raw
      Writes 90 days of synthetic PM2.5/PM10 and weather to data/raw/ so the full
      pipeline (train + predict) can run offline. Pollution follows Mumbai-like
      patterns: morning and late-night peaks, an afternoon sea-breeze dip, and
      lower levels when the wind is strong.

The numbers are SIMULATED. Use fetch_aqi.py and fetch_weather.py for real data.
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import HISTORY_DAYS, RAW_DIR, STATIONS

# Relative pollution level per station (industrial east side and port areas run higher)
STATION_FACTOR = {
    "Bandra Kurla Complex": 1.15, "Colaba": 0.72, "Worli": 0.88, "Chakala, Andheri East": 1.10,
    "Borivali East": 0.93, "Mazgaon": 1.42, "Sion": 1.08, "Powai": 0.81, "Chembur": 1.18, "Deonar": 1.58,
}


def synthetic_raw(days=HISTORY_DAYS, seed=7):
    rng = np.random.default_rng(seed)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    ts = pd.date_range(now - timedelta(days=days), now + timedelta(days=4), freq="h")
    hour = ts.hour.to_numpy()

    # Weather: sea breeze strengthens in the afternoon; humid nights
    wind = 9 + 7 * np.exp(-((hour - 15) ** 2) / 12) + rng.normal(0, 1.5, len(ts))
    wind += 3 * np.sin(np.arange(len(ts)) / (24 * 5))          # multi-day weather systems
    hum = 68 + 12 * np.cos((hour - 5) / 24 * 2 * np.pi) + rng.normal(0, 3, len(ts))
    temp = 29 + 4 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.8, len(ts))
    wdir = np.where((hour >= 11) & (hour <= 19), 270, 70) + rng.normal(0, 25, len(ts))  # sea vs land breeze
    weather = pd.DataFrame({"ts": ts, "temp_c": temp.round(1), "humidity_pct": hum.clip(30, 100).round(),
                            "wind_kmh": wind.clip(0.5).round(1), "wind_dir_deg": (wdir % 360).round()})

    observed = ts <= now
    diurnal = 1 + 0.35 * np.exp(-((hour - 9) ** 2) / 8) + 0.45 * np.exp(-(((hour + 1) % 24) ** 2) / 10) \
        - 0.25 * np.exp(-((hour - 15) ** 2) / 9)
    season = 1 + 0.25 * (np.arange(len(ts)) / len(ts))          # pollution rises into winter
    wind_effect = np.exp(-0.035 * (wind - 10))
    rows = []
    for st in STATIONS:
        f = STATION_FACTOR.get(st["station"], 1.0)
        noise = np.zeros(len(ts))
        for i in range(1, len(ts)):
            noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.06)
        level = 52 * f * diurnal * season * wind_effect * np.exp(noise)
        pm25 = level * rng.normal(1, 0.05, len(ts))
        pm10 = level * 1.9 * rng.normal(1, 0.06, len(ts))
        df = pd.DataFrame({"station": st["station"], "ts": ts, "pm25": pm25.round(1), "pm10": pm10.round(1)})
        rows.append(df[observed])
    aqi = pd.concat(rows, ignore_index=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    aqi.to_csv(RAW_DIR / "aqi_hourly.csv", index=False)
    weather.to_csv(RAW_DIR / "weather_hourly.csv", index=False)
    print(f"Synthetic raw data written to {RAW_DIR} ({len(aqi):,} AQI rows, {len(weather):,} weather rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", action="store_true", help="write synthetic raw history to data/raw/")
    p.add_argument("--days", type=int, default=HISTORY_DAYS)
    a = p.parse_args()
    synthetic_raw(a.days)
