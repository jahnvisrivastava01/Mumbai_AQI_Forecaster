# Mumbai AQI Forecaster

Forecasts air quality for **every government monitoring station across the Mumbai region** (Mumbai city and suburbs, Navi Mumbai, Thane and Mira-Bhayandar) up to **72 hours ahead**. It explains what drives each forecast, finds the best time to go outside, and shows everything on a live dashboard with health advice by area.

**Live demo:** https://mumbai-aqi-forecaster.onrender.com

![Dashboard screenshot](docs/screenshot.png)

## What it does

- **Automatic station discovery**: finds every active reference-grade monitor (CPCB / MPCB / IITM / BMC) inside the region's bounding box via the OpenAQ API. New stations are picked up without code changes; low-cost sensors and stations that stopped reporting are skipped.
- **Live data pipeline**: pulls hourly PM2.5 and PM10 for each station, plus weather history and forecast from Open-Meteo.
- **Indian AQI calculation**: computes the CPCB National AQI from 24-hour average pollutant concentrations.
- **72-hour forecasting**: one LightGBM model predicts every horizon from 1 to 72 hours (direct multi-horizon), using recent AQI, the weather forecast, time of day and festival dates.
- **Upwind neighbour features**: each station's forecast also uses nearby stations' readings, weighted by distance and by how directly the wind blows from them. Pollution over Deonar now tends to reach Chembur next. An ablation model trained without these features is reported alongside, so their effect is measured, not assumed.
- **Forecast range**: 10th and 90th percentile quantile models give an uncertainty band.
- **Honest evaluation**: time-based train/test split (most recent 14 days held out), compared against a persistence baseline. MAE and RMSE reported per horizon.
- **Explainability**: SHAP values show which features move the forecast most.
- **Dashboard**: best time to go outside, station map, forecast vs measured chart, weather inputs, SHAP chart, an hour-by-weekday pollution heatmap, health alerts and model comparison, with a light/dark toggle. Plain HTML + D3.
- **Automated daily refresh**: a GitHub Actions workflow re-fetches data and retrains every morning at 6 AM IST; Render redeploys the site automatically.

## Results

Evaluated on the most recent 14 days, held out from training. Error is in AQI points (lower is better).

| Model | 24h MAE | 48h MAE | 72h MAE |
|---|---|---|---|
| Persistence baseline (same hour, previous day) | _fill in_ | _fill in_ | _fill in_ |
| LightGBM without neighbour data | _fill in_ | _fill in_ | _fill in_ |
| **LightGBM + weather + neighbours** | **_fill in_** | **_fill in_** | **_fill in_** |

Fill these in from the table printed at the end of `python -m pipeline.run_all`. The live dashboard's **Model comparison** card always shows the latest numbers.

## Architecture

```
OpenAQ API ──┐
             ├─► data/raw/*.csv ─► features.py ─► train.py ─► models/ ─► predict.py ─► dashboard/data/*.csv ─► dashboard (D3)
Open-Meteo ──┘                      (lags, weather,  (LightGBM,              (72h forecast,
                                     neighbours)      quantiles, SHAP)         stations, metrics,
                                                                               hourly profile)

GitHub Actions (daily) runs the pipeline and commits dashboard/data ─► Render redeploys the static site
```

| File | Purpose |
|---|---|
| `pipeline/config.py` | Region bounding box, fallback station list, paths, forecast settings, festival dates |
| `pipeline/fetch_aqi.py` | Discovers stations and fetches hourly PM2.5 / PM10 from OpenAQ v3 |
| `pipeline/fetch_weather.py` | Weather history + 4-day forecast from Open-Meteo |
| `pipeline/aqi.py` | CPCB AQI breakpoints, categories, health advice |
| `pipeline/features.py` | Lag, weather, calendar and upwind-neighbour features; builds (station, origin, horizon) rows |
| `pipeline/train.py` | Baseline, LightGBM point + quantile models, neighbour ablation, metrics, SHAP |
| `pipeline/predict.py` | Live forecast and dashboard export |
| `pipeline/generate_sample.py` | Synthetic data for offline runs |
| `pipeline/run_all.py` | Runs everything in order |
| `dashboard/` | Static site: `index.html`, `style.css`, `app.js`, `data/` |
| `.github/workflows/refresh.yml` | Daily data refresh and retraining |

## Run it locally

```bash
git clone https://github.com/jahnvisrivastava01/Mumbai_AQI_Forecaster.git
cd Mumbai_AQI_Forecaster
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option A: offline, synthetic data (no API key needed)
python -m pipeline.run_all --offline

# Option B: real data
cp .env.example .env        # then paste your OpenAQ key into .env
python -m pipeline.run_all

# Reuse already-downloaded data and only retrain
python -m pipeline.run_all --skip-fetch

# View the dashboard
cd dashboard && python -m http.server 8000
# open http://localhost:8000
```

Get a free OpenAQ API key at https://explore.openaq.org (sign up, then account settings).

The dashboard shows a **Sample data** or **Live data** badge so it's always clear which you're looking at.

## Deployment

- **Data refresh**: GitHub Actions runs `.github/workflows/refresh.yml` every day at 6:00 AM IST. It fetches the latest readings, retrains and commits the new `dashboard/data/` files. Add your key under **Settings → Secrets and variables → Actions** as `OPENAQ_API_KEY`, and allow **Read and write** workflow permissions under **Settings → Actions → General**.
- **Hosting**: Render static site with **Publish Directory** `dashboard` and an empty build command. It redeploys automatically on every commit, including the daily data commit.

## Model details

- **Target**: CPCB AQI at time t+h, from 24-hour rolling PM2.5 / PM10 means.
- **Approach**: the model predicts the correction to the persistence forecast (same hour, previous day). Tree models can't extrapolate to AQI levels they never saw in training, but corrections stay in a stable range as pollution rises into winter.
- **Neighbour features**: for each station, the inverse-distance-weighted AQI and 3-hour trend of other stations within 15 km, plus an upwind version where each neighbour's weight is multiplied by the cosine between the wind direction and the bearing to that neighbour (only stations the wind blows from count).
- **Other features**: AQI now and lags (1h, 3h, 24h), 6h and 24h means, 3h trend, raw PM2.5 / PM10, horizon, target hour and weekday, wind speed and direction (land vs sea breeze), humidity, temperature, festival flag, station.
- **Training**: forecast origins every 3 hours, early stopping on the last week of the training period.
- **Evaluation**: rows whose target falls in the last 14 days are held out entirely.
- **Best time to go outside**: the forecast AQI shaped by each area's typical hour-by-hour PM2.5 pattern, since AQI itself is a 24-hour average and barely changes within a day.

## Changing the region

Edit `REGION_BBOX` in `pipeline/config.py` (min longitude, min latitude, max longitude, max latitude) and rerun. For example, `(72.76, 18.88, 73.25, 19.45)` also takes in Vasai-Virar, Kalyan and Dombivli. The map, dropdown and tables adapt to whatever stations are found.

## Ideas to extend

- Add NO2, O3 and CO sub-indices for the full CPCB AQI
- Fetch station-level weather instead of one city point
- Longer training history from Kaggle's CPCB dataset (2015–2020)
- Alerts by email or Telegram when the forecast crosses Poor
- A FastAPI endpoint serving forecasts for any station
- Track forecast accuracy over time (yesterday's forecast vs what happened)

## License

MIT © Jahnvi Srivastava
