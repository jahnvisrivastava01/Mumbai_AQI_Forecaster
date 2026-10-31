"""Fetch hourly weather (history + 4-day forecast) for Mumbai from Open-Meteo.

Open-Meteo is free and needs no API key. One call returns past days and the
forecast, which the model uses as inputs for future hours.

Output: data/raw/weather_hourly.csv  (ts, temp_c, humidity_pct, wind_kmh, wind_dir_deg)
"""
import pandas as pd
import requests

from .config import CITY_CENTER, HISTORY_DAYS, RAW_DIR, TIMEZONE

URL = "https://api.open-meteo.com/v1/forecast"


def main(days=HISTORY_DAYS):
    lat, lon = CITY_CENTER
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "past_days": min(days, 92), "forecast_days": 4,
        "timezone": TIMEZONE, "wind_speed_unit": "kmh",
    }
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(h["time"]),
        "temp_c": h["temperature_2m"],
        "humidity_pct": h["relative_humidity_2m"],
        "wind_kmh": h["wind_speed_10m"],
        "wind_dir_deg": h["wind_direction_10m"],
    })
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "weather_hourly.csv", index=False)
    print(f"Saved {len(df)} rows to {RAW_DIR / 'weather_hourly.csv'}")


if __name__ == "__main__":
    main()
