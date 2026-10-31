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

NEIGHBOUR_RADIUS_KM = 15
NEIGHBOUR_FEATURES = ["nbr_aqi", "nbr_change3", "upwind_aqi", "upwind_gap"]

FEATURES = [
    "station_id", "horizon", "target_hour", "target_dow",
    "aqi_now", "aqi_lag1", "aqi_lag3", "aqi_lag24", "aqi_mean6", "aqi_mean24", "aqi_change3",
    "pm25_now", "pm10_now", "persist",
    "wind_kmh", "wind_from_east", "wind_from_north", "humidity_pct", "temp_c", "event",
    *NEIGHBOUR_FEATURES,
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
    "nbr_aqi": "Nearby stations' AQI", "nbr_change3": "Nearby stations' 3-hour trend",
    "upwind_aqi": "Upwind stations' AQI", "upwind_gap": "Upwind minus own AQI",
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
    df = add_neighbour_features(df, wx, load_stations())
    return df, wx


def add_neighbour_features(df, wx, stations, radius_km=NEIGHBOUR_RADIUS_KM):
    """Spatial features from other stations within `radius_km`, weighted by 1/distance.

    nbr_aqi      distance-weighted AQI of nearby stations right now
    nbr_change3  their 3-hour trend (pollution building up or clearing nearby)
    upwind_aqi   same, but only stations the wind is blowing FROM, weighted by how
                 directly upwind they are; pollution there tends to arrive here next
    upwind_gap   upwind_aqi minus this station's own AQI
    """
    st = stations.drop_duplicates("station").set_index("station")
    names = [n for n in sorted(df["station"].unique()) if n in st.index]
    for col in NEIGHBOUR_FEATURES:
        df[col] = np.nan
    if len(names) < 2:
        return df

    wide = df.pivot_table(index="ts", columns="station", values="aqi")[names]
    A = wide.to_numpy()                                  # time x station
    change3 = A - np.vstack([np.full((3, A.shape[1]), np.nan), A[:-3]])

    lat = np.radians(st.loc[names, "lat"].astype(float).to_numpy())
    lon = np.radians(st.loc[names, "lon"].astype(float).to_numpy())
    dlat, dlon = lat[None, :] - lat[:, None], lon[None, :] - lon[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))              # km, [station, neighbour]
    bearing = np.arctan2(np.sin(dlon) * np.cos(lat[None, :]),
                         np.cos(lat[:, None]) * np.sin(lat[None, :])
                         - np.sin(lat[:, None]) * np.cos(lat[None, :]) * np.cos(dlon))
    idw = np.where((dist > 0) & (dist <= radius_km), 1 / np.maximum(dist, 0.5), 0.0)

    def weighted(values, weights):
        ok = ~np.isnan(values)
        v = np.where(ok, values, 0.0)
        if weights.ndim == 2:                            # same weights at every hour
            num, den = v @ weights.T, ok.astype(float) @ weights.T
        else:                                            # weights change hourly with the wind
            num = np.einsum("tsn,tn->ts", weights, v)
            den = np.einsum("tsn,tn->ts", weights, ok.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    wind_from = np.radians(wx["wind_dir_deg"].reindex(wide.index, method="nearest").to_numpy())
    align = np.cos(bearing[None, :, :] - wind_from[:, None, None])   # 1 = neighbour directly upwind
    up_w = idw[None, :, :] * np.clip(np.nan_to_num(align), 0, None)

    feats = {
        "nbr_aqi": weighted(A, idw),
        "nbr_change3": weighted(change3, idw),
        "upwind_aqi": weighted(A, up_w),
    }
    feats["upwind_gap"] = feats["upwind_aqi"] - A
    long = pd.concat([
        pd.DataFrame(v, index=wide.index, columns=names).stack(future_stack=True).rename(k)
        for k, v in feats.items()
    ], axis=1).reset_index()
    long.columns = ["ts", "station", *feats.keys()]
    df = df.drop(columns=NEIGHBOUR_FEATURES).merge(long, on=["ts", "station"], how="left")
    return df


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
    for col in NEIGHBOUR_FEATURES:
        o[col] = s[col]
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