"""Run the whole pipeline: fetch -> train -> forecast -> export to dashboard.

  python -m pipeline.run_all            real data (needs OPENAQ_API_KEY)
  python -m pipeline.run_all --offline  synthetic data, no API key or network needed
  python -m pipeline.run_all --skip-fetch   reuse data/raw/ from the last fetch
"""
import argparse

from . import fetch_aqi, fetch_weather, generate_sample, predict, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true", help="use synthetic data instead of the APIs")
    p.add_argument("--skip-fetch", action="store_true", help="reuse files already in data/raw/")
    a = p.parse_args()

    if a.offline:
        print("1/3  Generating synthetic data")
        generate_sample.synthetic_raw()
    elif not a.skip_fetch:
        print("1/3  Fetching air quality (OpenAQ)")
        fetch_aqi.main()
        print("     Fetching weather (Open-Meteo)")
        fetch_weather.main()
    print("2/3  Training models")
    train.main()
    print("3/3  Forecasting and exporting")
    predict.main(source="sample" if a.offline else "live")


if __name__ == "__main__":
    main()
