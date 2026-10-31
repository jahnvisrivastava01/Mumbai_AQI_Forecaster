"""Project-wide settings: stations, paths, and model constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
DASHBOARD_DATA = ROOT / "dashboard" / "data"

TIMEZONE = "Asia/Kolkata"
CITY_CENTER = (19.0760, 72.8777)

# Region searched for monitoring stations: (min lon, min lat, max lon, max lat).
# Covers Mumbai city and suburbs, Navi Mumbai, Thane and Mira-Bhayandar.
REGION_BBOX = (72.76, 18.88, 73.12, 19.32)

# Fallback station list, used for offline sample data and if OpenAQ discovery finds nothing.
# Live runs discover stations automatically and write data/raw/stations.csv instead.
STATIONS = [
    # South and central Mumbai
    {"station": "Colaba", "area": "Colaba", "lat": 18.9067, "lon": 72.8147},
    {"station": "Navy Nagar, Colaba", "area": "Navy Nagar", "lat": 18.8970, "lon": 72.8130},
    {"station": "Mazgaon", "area": "Mazgaon", "lat": 18.9668, "lon": 72.8436},
    {"station": "Byculla", "area": "Byculla", "lat": 18.9790, "lon": 72.8330},
    {"station": "Siddharth Nagar, Worli", "area": "Siddharth Nagar", "lat": 19.0000, "lon": 72.8150},
    {"station": "Worli", "area": "Worli", "lat": 19.0176, "lon": 72.8180},
    {"station": "Sewri", "area": "Sewri", "lat": 19.0005, "lon": 72.8570},
    {"station": "Sion", "area": "Sion", "lat": 19.0470, "lon": 72.8640},
    # Western suburbs
    {"station": "Bandra", "area": "Bandra", "lat": 19.0627, "lon": 72.8350},
    {"station": "Bandra Kurla Complex", "area": "BKC", "lat": 19.0656, "lon": 72.8644},
    {"station": "Vile Parle West", "area": "Vile Parle", "lat": 19.1000, "lon": 72.8370},
    {"station": "Chakala, Andheri East", "area": "Andheri East", "lat": 19.1110, "lon": 72.8606},
    {"station": "Malad West", "area": "Malad", "lat": 19.1870, "lon": 72.8380},
    {"station": "Kandivali East", "area": "Kandivali", "lat": 19.2050, "lon": 72.8710},
    {"station": "Borivali East", "area": "Borivali", "lat": 19.2321, "lon": 72.8691},
    {"station": "Mira-Bhayandar", "area": "Mira Road", "lat": 19.2810, "lon": 72.8690},
    # Eastern suburbs
    {"station": "Kurla", "area": "Kurla", "lat": 19.0728, "lon": 72.8826},
    {"station": "Chembur", "area": "Chembur", "lat": 19.0522, "lon": 72.9005},
    {"station": "Deonar", "area": "Deonar", "lat": 19.0490, "lon": 72.9200},
    {"station": "Ghatkopar", "area": "Ghatkopar", "lat": 19.0860, "lon": 72.9080},
    {"station": "Powai", "area": "Powai", "lat": 19.1176, "lon": 72.9060},
    {"station": "Khindipada, Bhandup West", "area": "Bhandup", "lat": 19.1460, "lon": 72.9330},
    {"station": "Mulund West", "area": "Mulund", "lat": 19.1720, "lon": 72.9420},
    # Thane and Navi Mumbai
    {"station": "Kasarvadavali, Thane", "area": "Thane", "lat": 19.2580, "lon": 72.9690},
    {"station": "Airoli", "area": "Airoli", "lat": 19.1590, "lon": 72.9986},
    {"station": "Mahape", "area": "Mahape", "lat": 19.1130, "lon": 73.0283},
    {"station": "Kopripada, Vashi", "area": "Vashi", "lat": 19.0815, "lon": 73.0094},
    {"station": "Nerul", "area": "Nerul", "lat": 19.0330, "lon": 73.0197},
    {"station": "Kharghar", "area": "Kharghar", "lat": 19.0473, "lon": 73.0699},
]

# Forecast settings
HORIZON_HOURS = 72
ORIGIN_STEP_HOURS = 3      # sample a forecast origin every N hours when building training rows
TEST_DAYS = 14             # most recent days held out for evaluation
HISTORY_DAYS = 90          # how much history to fetch (Open-Meteo forecast API allows up to 92 past days)

# Festival / event dates that change emissions (fireworks, traffic). Extend as needed.
EVENT_DATES = [
    # Diwali 2025
    "2025-10-20", "2025-10-21", "2025-10-22", "2025-10-23",
    # Diwali 2026
    "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10",
    # New Year's Eve
    "2025-12-31", "2026-01-01", "2026-12-31",
]