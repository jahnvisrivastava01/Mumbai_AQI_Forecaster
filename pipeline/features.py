"""Feature engineering for direct multi-horizon AQI forecasting.

Each training row is (station, forecast origin t, horizon h): features describe
what is known at time t (recent AQI) plus the weather forecast and calendar at
the target time t+h. One model learns every horizon from 1 to 72 hours, with h
as a feature. This avoids the error build-up of recursive forecasting.
"""
import numpy as np
import pandas as pd

from .aqi import add_aqi
from .config import EVENT_DATES, HORIZON_HOURS, RAW_DIR, STATIONS


def load_stations() -> pd.DataFrame:
    """Stations found by the last fetch (data/raw/stations.csv), else the config list."""
    path = RAW_DIR / "stations.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(STATIONS)


FEATURES = [
    "station_id", "horizon", "target_hour", "target_dow",
    "aqi_now", "aqi_lag1", "aqi_lag3", "aqi_lag24", "aqi_mean6", "aqi_mean24", "aqi_change3",
    "pm25_now", "pm10_now", "persist",
    "wind_kmh", "wind_from_east", "wind_from_north", "humidity_pct", "temp_c", "event",
]

# Plain-language names for the dashboard's feature-importance chart
FEATURE_LABELS = {
    "station_id": "Station location", "horizon": "Hours ahead", "target_hour": "Hour of day",
    "target_dow": "Day of week", "aqi_now": "AQI at forecast time", "aqi_lag1": "AQI 1 hour earlier",
    "aqi_lag3": "AQI 3 hours earlier", "aqi_lag24": "AQI 24 hours earlier",
    "aqi_mean6": "6-hour average AQI", "aqi_mean24": "24-hour average AQI",
    "aqi_change3": "3-hour AQI trend", "pm25_now": "PM2.5 now", "pm10_now": "PM10 now",
    "persist": "Same hour, previous day", "wind_kmh": "Wind speed",
    "wind_from_east": "Land breeze (wind from east)", "wind_from_north": "Wind from north",
    "humidity_pct": "Relative humidity", "temp_c": "Temperature", "event": "Festival or event flag",
}


def load_raw():
    aqi = pd.read_csv(RAW_DIR / "aqi_hourly.csv", parse_dates=["ts"])
    wx = pd.read_csv(RAW_DIR / "weather_hourly.csv", parse_dates=["ts"])
    return aqi, wx


def prepare(aqi_raw: pd.DataFrame, wx: pd.DataFrame):
    """Regular hourly grid per station, small gaps filled, AQI computed."""
    frames = []
    for name, g in aqi_raw.groupby("station"):
        g = g.groupby("ts")[["pm25", "pm10"]].mean()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(idx).interpolate(limit=3, limit_area="inside")
        g.index.name = "ts"
        g = g.reset_index()
        g["station"] = name
        frames.append(g)
    df = add_aqi(pd.concat(frames, ignore_index=True))

    wx = wx.drop_duplicates("ts").set_index("ts").sort_index()
    wx = wx.reindex(pd.date_range(wx.index.min(), wx.index.max(), freq="h")).interpolate(limit=6)
    rad = np.deg2rad(wx["wind_dir_deg"])
    wx["wind_from_east"] = np.sin(rad)   # +1 when wind blows from the east (land breeze)
    wx["wind_from_north"] = np.cos(rad)
    wx.index.name = "ts"
    return df, wx


def _origin_features(s: pd.DataFrame) -> pd.DataFrame:
    """s: one station, indexed by ts on a regular hourly grid."""
    a = s["aqi"]
    o = pd.DataFrame(index=s.index)
    o["aqi_now"] = a
    o["aqi_lag1"] = a.shift(1)
    o["aqi_lag3"] = a.shift(3)
    o["aqi_lag24"] = a.shift(24)
    o["aqi_mean6"] = a.rolling(6, min_periods=4).mean()
    o["aqi_mean24"] = a.rolling(24, min_periods=16).mean()
    o["aqi_change3"] = a - a.shift(3)
    o["pm25_now"] = s["pm25"]
    o["pm10_now"] = s["pm10"]
    return o


def make_rows(df, wx, station_names, origins="sampled", step=3, horizons=range(1, HORIZON_HOURS + 1)):
    """Build (origin, horizon) rows.

    origins="sampled": every `step` hours (for training).
    origins="latest":  the last hour with an AQI value per station (for the live forecast).
    station_names: fixed order used to encode the station as a category (saved with the model).
    """
    events = set(pd.to_datetime(EVENT_DATES).date)
    out = []
    for name, s in df.groupby("station"):
        s = s.set_index("ts").sort_index()
        o = _origin_features(s).dropna(subset=["aqi_now"])
        if o.empty:
            continue
        o = o.iloc[::step] if origins == "sampled" else o.iloc[[-1]]
        aqi_by_ts = s["aqi"]
        for h in horizons:
            r = o.copy()
            r["origin"] = r.index
            r["target_ts"] = r.index + pd.Timedelta(hours=h)
            r["horizon"] = h
            back = 24 * int(np.ceil(h / 24))
            r["persist"] = aqi_by_ts.reindex(r["target_ts"] - pd.Timedelta(hours=back)).to_numpy()
            r["y"] = aqi_by_ts.reindex(r["target_ts"]).to_numpy()
            w = wx.reindex(r["target_ts"], method="nearest", tolerance=pd.Timedelta(hours=3))
            for col in ("wind_kmh", "wind_from_east", "wind_from_north", "humidity_pct", "temp_c"):
                r[col] = w[col].to_numpy()
            r["target_hour"] = r["target_ts"].dt.hour
            r["target_dow"] = r["target_ts"].dt.dayofweek
            r["event"] = r["target_ts"].dt.date.isin(events).astype(int)
            r["station"] = name
            out.append(r.reset_index(drop=True))
    rows = pd.concat(out, ignore_index=True)
    rows["station_id"] = pd.Categorical(rows["station"], categories=station_names).codes
    return rows