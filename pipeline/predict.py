"""Make the live 72-hour forecast and export everything the dashboard reads.

Writes to dashboard/data/:
  aqi_hourly.csv         last 72h measured + next 72h forecast per station and city average
  aqi_stations.csv       current reading, category and advice per station
  aqi_drivers.csv        SHAP feature importance
  aqi_model_metrics.csv  model comparison
  meta.json              data source and generation time
"""
import json
import shutil
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from .aqi import advice, category
from .config import DASHBOARD_DATA, MODEL_DIR
from .features import FEATURES, load_raw, load_stations, make_rows, prepare
from .train import baseline

PAST_HOURS = 72


def main(source="live"):
    aqi_raw, wx = load_raw()
    df, wx = prepare(aqi_raw, wx)
    station_names = json.loads((MODEL_DIR / "train_info.json").read_text())["stations"]
    rows = make_rows(df, wx, station_names, origins="latest")

    X = rows[FEATURES]
    base = baseline(rows)  # models predict the correction to the persistence forecast
    point = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_point.txt")).predict(X) + base
    lo = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_q10.txt")).predict(X) + base
    hi = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_q90.txt")).predict(X) + base
    rows["aqi"] = np.clip(point, 0, 500)
    rows["aqi_low"] = np.clip(np.minimum(lo, rows["aqi"]), 0, 500)
    rows["aqi_high"] = np.clip(np.maximum(hi, rows["aqi"]), 0, 500)
    fc = rows.rename(columns={"target_ts": "ts"})[["station", "ts", "aqi", "aqi_low", "aqi_high"]]
    fc["kind"] = "forecast"

    past = []
    for name, s in df.dropna(subset=["aqi"]).groupby("station"):
        past.append(s.sort_values("ts").tail(PAST_HOURS)[["station", "ts", "aqi"]])
    past = pd.concat(past)
    past["kind"] = "actual"

    hourly = pd.concat([past, fc], ignore_index=True)
    w = wx[["wind_kmh", "humidity_pct"]].reindex(hourly["ts"], method="nearest")
    hourly["wind_kmh"] = w["wind_kmh"].round(1).to_numpy()
    hourly["humidity_pct"] = w["humidity_pct"].round().to_numpy()

    city = (hourly.groupby(["ts", "kind"], as_index=False)
            .agg(aqi=("aqi", "mean"), aqi_low=("aqi_low", "mean"), aqi_high=("aqi_high", "mean"),
                 wind_kmh=("wind_kmh", "first"), humidity_pct=("humidity_pct", "first")))
    city["station"] = "City average"
    hourly = pd.concat([hourly, city], ignore_index=True)
    for col in ("aqi", "aqi_low", "aqi_high"):
        hourly[col] = hourly[col].round().astype("Int64")
    hourly.loc[hourly["kind"] == "actual", ["aqi_low", "aqi_high"]] = pd.NA
    hourly["ts"] = pd.to_datetime(hourly["ts"]).dt.strftime("%Y-%m-%dT%H:00")
    hourly = hourly.sort_values(["station", "ts"])[
        ["station", "ts", "kind", "aqi", "aqi_low", "aqi_high", "wind_kmh", "humidity_pct"]]

    latest = df.dropna(subset=["aqi"]).sort_values("ts").groupby("station").tail(1).set_index("station")
    st_rows = []
    for st in load_stations().to_dict("records"):
        if st["station"] not in latest.index:
            continue
        r = latest.loc[st["station"]]
        a = int(round(r["aqi"]))
        # Some stations measure only PM2.5 or only PM10; leave the missing one blank
        pm25 = None if pd.isna(r["pm25_24h"]) else int(round(r["pm25_24h"]))
        pm10 = None if pd.isna(r["pm10_24h"]) else int(round(r["pm10_24h"]))
        dominant = r["dominant"] if pm25 is not None and pm10 is not None else ("PM2.5" if pm10 is None else "PM10")
        st_rows.append({**st, "aqi": a, "pm25": pm25, "pm10": pm10,
                        "dominant": dominant, "category": category(a), "advice": advice(a)})

    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(DASHBOARD_DATA / "aqi_hourly.csv", index=False)
    st_df = pd.DataFrame(st_rows)
    st_df[["pm25", "pm10"]] = st_df[["pm25", "pm10"]].astype("Int64")   # whole numbers, blank if missing
    st_df.to_csv(DASHBOARD_DATA / "aqi_stations.csv", index=False)
    shutil.copy(MODEL_DIR / "shap.csv", DASHBOARD_DATA / "aqi_drivers.csv")
    shutil.copy(MODEL_DIR / "metrics.csv", DASHBOARD_DATA / "aqi_model_metrics.csv")
    (DASHBOARD_DATA / "meta.json").write_text(json.dumps({
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "latest_reading": str(df.dropna(subset=["aqi"])["ts"].max()),
    }, indent=2))
    print(f"Dashboard data written to {DASHBOARD_DATA}")


if __name__ == "__main__":
    main()