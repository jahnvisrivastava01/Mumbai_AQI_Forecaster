"""Indian National AQI (CPCB) calculation from PM2.5 and PM10.

CPCB computes sub-indices from 24-hour average concentrations (µg/m³) and the
overall AQI is the maximum sub-index. This project uses PM2.5 and PM10, which
dominate Mumbai's AQI on almost every day.
"""
import numpy as np
import pandas as pd

# (conc_low, conc_high, aqi_low, aqi_high)
PM25_BREAKPOINTS = [
    (0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
    (90, 120, 200, 300), (120, 250, 300, 400), (250, 500, 400, 500),
]
PM10_BREAKPOINTS = [
    (0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
    (250, 350, 200, 300), (350, 430, 300, 400), (430, 600, 400, 500),
]

CATEGORIES = [
    (50, "Good", "No precautions needed"),
    (100, "Satisfactory", "Sensitive groups: limit long outdoor exertion"),
    (200, "Moderate", "Asthma and heart patients: reduce outdoor activity"),
    (300, "Poor", "Everyone: cut outdoor exercise; masks for sensitive groups"),
    (400, "Very poor", "Avoid outdoor activity; use N95 outdoors"),
    (10_000, "Severe", "Stay indoors; run air purifiers"),
]


def sub_index(conc, breakpoints):
    """Piecewise-linear sub-index for an array of concentrations."""
    conc = np.asarray(conc, dtype=float)
    out = np.full(conc.shape, np.nan)
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        mask = (conc >= c_lo) & (conc <= c_hi) & np.isnan(out)
        out[mask] = a_lo + (conc[mask] - c_lo) * (a_hi - a_lo) / (c_hi - c_lo)
    out[conc > breakpoints[-1][1]] = 500
    return out


def indian_aqi(pm25_24h, pm10_24h):
    """Return (aqi, dominant_pollutant) arrays."""
    s25 = sub_index(pm25_24h, PM25_BREAKPOINTS)
    s10 = sub_index(pm10_24h, PM10_BREAKPOINTS)
    both = np.vstack([np.nan_to_num(s25, nan=-1), np.nan_to_num(s10, nan=-1)])
    aqi = both.max(axis=0)
    aqi[aqi < 0] = np.nan
    dominant = np.where(both[0] >= both[1], "PM2.5", "PM10")
    return aqi, dominant


def category(aqi):
    for limit, name, _ in CATEGORIES:
        if aqi <= limit:
            return name
    return "Severe"


def advice(aqi):
    for limit, _, text in CATEGORIES:
        if aqi <= limit:
            return text
    return CATEGORIES[-1][2]


def add_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Add 24h rolling means and AQI to an hourly frame with station, ts, pm25, pm10."""
    df = df.sort_values(["station", "ts"]).copy()
    g = df.groupby("station", group_keys=False)
    df["pm25_24h"] = g["pm25"].transform(lambda s: s.rolling(24, min_periods=16).mean())
    df["pm10_24h"] = g["pm10"].transform(lambda s: s.rolling(24, min_periods=16).mean())
    df["aqi"], df["dominant"] = indian_aqi(df["pm25_24h"], df["pm10_24h"])
    return df
