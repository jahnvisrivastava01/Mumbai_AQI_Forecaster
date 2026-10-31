"""Fetch hourly PM2.5 and PM10 for every government monitor in the Mumbai region.

Stations are discovered automatically from OpenAQ (v3 API) inside the region's
bounding box (config.REGION_BBOX): Mumbai city and suburbs, Navi Mumbai, Thane
and Mira-Bhayandar. Only reference-grade monitors (CPCB / MPCB / IITM / BMC) are
used, not low-cost sensors. If discovery finds nothing, it falls back to the
hand-listed stations in config.STATIONS.

A free API key is required: sign up at https://explore.openaq.org and put it in
a .env file as OPENAQ_API_KEY=your_key

Outputs:
  data/raw/stations.csv    station, area, lat, lon   (the stations found)
  data/raw/aqi_hourly.csv  station, ts, pm25, pm10   (ts in IST)
"""
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import HISTORY_DAYS, RAW_DIR, REGION_BBOX, STATIONS, TIMEZONE

BASE = "https://api.openaq.org/v3"
SEARCH_RADIUS_M = 3000
ACTIVE_WITHIN_DAYS = 14      # skip stations that stopped reporting


def _session():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise SystemExit("OPENAQ_API_KEY is not set. Add it to .env or your environment.")
    s = requests.Session()
    s.headers["X-API-Key"] = key
    return s


def _get(s, path, params=None, attempts=4):
    """GET with retries on rate limits (429), OpenAQ server errors (5xx) and timeouts."""
    for attempt in range(attempts):
        try:
            r = s.get(f"{BASE}{path}", params=params, timeout=60)
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt * 3)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == attempts - 1:
                r.raise_for_status()
            time.sleep(2 ** attempt * 3)
            continue
        r.raise_for_status()
        return r.json()


def _pm_sensors(loc):
    found = {}
    for sensor in loc.get("sensors", []):
        name = (sensor.get("parameter") or {}).get("name")
        if name in ("pm25", "pm10") and name not in found:
            found[name] = sensor["id"]
    return found


def _clean_name(raw):
    """'Navy Nagar-Colaba, Mumbai - IITM' -> ('Navy Nagar-Colaba, Mumbai', 'Navy Nagar-Colaba')."""
    station = re.split(r"\s+-\s+", raw or "")[0].strip() or "Unnamed station"
    area = station.split(",")[0].strip()
    return station, area


def _is_active(loc, now):
    last = ((loc.get("datetimeLast") or {}).get("utc"))
    if not last:
        return True
    return pd.Timestamp(last) >= pd.Timestamp(now - timedelta(days=ACTIVE_WITHIN_DAYS))


def discover_stations(s, now):
    """All active reference monitors with PM2.5 or PM10 in the region's bounding box."""
    params = {"bbox": ",".join(str(v) for v in REGION_BBOX), "limit": 1000, "monitor": "true"}
    try:
        data = _get(s, "/locations", params)
    except requests.HTTPError:
        params.pop("monitor")            # older API versions reject this filter
        data = _get(s, "/locations", params)
    stations, seen = [], {}
    for loc in data.get("results", []):
        if loc.get("isMonitor") is False:
            continue
        sensors = _pm_sensors(loc)
        if not sensors or not _is_active(loc, now):
            continue
        station, area = _clean_name(loc.get("name"))
        if station in seen:              # keep names unique
            seen[station] += 1
            station = f"{station} ({seen[station]})"
        else:
            seen[station] = 1
        c = loc.get("coordinates") or {}
        stations.append({"station": station, "area": area, "lat": c.get("latitude"),
                         "lon": c.get("longitude"), "sensors": sensors})
    return stations


def stations_from_config(s):
    """Fallback: nearest OpenAQ location to each hand-listed station."""
    out, used = [], set()
    for st in STATIONS:
        data = _get(s, "/locations", {"coordinates": f"{st['lat']},{st['lon']}",
                                      "radius": SEARCH_RADIUS_M, "limit": 50})
        for loc in sorted(data.get("results", []), key=lambda l: l.get("distance") or 0):
            if loc["id"] in used:
                continue
            sensors = _pm_sensors(loc)
            if sensors:
                used.add(loc["id"])
                out.append({**st, "sensors": sensors})
                break
        else:
            print(f"  ! no PM sensors found near {st['station']}, skipping")
    return out


def sensor_hours(s, sensor_id, start, end, chunk_days=30):
    """Hourly values in 30-day windows. Each window is at most 720 rows, so it fits in one
    page of 1000 and never needs OpenAQ's paging, which often fails on long ranges."""
    rows = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        try:
            data = _get(s, f"/sensors/{sensor_id}/hours", {
                "datetime_from": cur.isoformat(), "datetime_to": nxt.isoformat(), "limit": 1000,
            })
            for r in data.get("results", []):
                rows.append({"ts_utc": r["period"]["datetimeFrom"]["utc"], "value": r["value"]})
        except requests.RequestException as e:
            print(f"    ! skipped {cur:%d %b}–{nxt:%d %b} for sensor {sensor_id}: {e.__class__.__name__}")
        cur = nxt
    df = pd.DataFrame(rows, columns=["ts_utc", "value"])
    return df.drop_duplicates("ts_utc", keep="last")


def main(days=HISTORY_DAYS):
    s = _session()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    stations = discover_stations(s, end)
    if stations:
        print(f"  Found {len(stations)} active monitoring stations in the Mumbai region")
    else:
        print("  No stations found in the region box; using the list in config.py")
        stations = stations_from_config(s)

    frames, kept = [], []
    for i, st in enumerate(stations, 1):
        merged = None
        for param, sid in st["sensors"].items():
            df = sensor_hours(s, sid, start, end)
            if df.empty:
                continue
            df = df.rename(columns={"value": param})
            merged = df if merged is None else merged.merge(df, on="ts_utc", how="outer")
        if merged is None or len(merged) < 24 * 7:
            print(f"  [{i}/{len(stations)}] {st['station']}: too little data, skipping")
            continue
        merged["station"] = st["station"]
        frames.append(merged)
        kept.append({k: st[k] for k in ("station", "area", "lat", "lon")})
        print(f"  [{i}/{len(stations)}] {st['station']}: {len(merged)} hours ({', '.join(st['sensors'])})")

    if not frames:
        raise SystemExit("No data fetched. Check your API key and network.")
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = (pd.to_datetime(out["ts_utc"], utc=True).dt.tz_convert(TIMEZONE)
                 .dt.tz_localize(None).dt.floor("h"))
    for col in ("pm25", "pm10"):
        if col not in out:
            out[col] = pd.NA
        out.loc[(out[col] < 0) | (out[col] > 1500), col] = pd.NA  # drop sensor glitches
    out = out[["station", "ts", "pm25", "pm10"]].sort_values(["station", "ts"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW_DIR / "aqi_hourly.csv", index=False)
    pd.DataFrame(kept).to_csv(RAW_DIR / "stations.csv", index=False)
    print(f"Saved {len(out):,} rows for {len(kept)} stations to {RAW_DIR}")


if __name__ == "__main__":
    main()