"""Train and evaluate the AQI forecaster.

Models compared on the last TEST_DAYS days (time-based split, no shuffling):
  * Persistence baseline: AQI at the same hour on the last observed day
  * Prophet (optional, if installed): seasonal model on the city-average series
  * LightGBM + weather: direct multi-horizon gradient boosting

Also trains 10th / 90th percentile LightGBM models for the forecast range,
and computes SHAP feature importance.

Outputs in models/: lgbm_point.txt, lgbm_q10.txt, lgbm_q90.txt, metrics.csv, shap.csv
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import MODEL_DIR, ORIGIN_STEP_HOURS, TEST_DAYS
from .features import FEATURE_LABELS, FEATURES, load_raw, make_rows, prepare

POINT_PARAMS = dict(objective="l2", n_estimators=700, learning_rate=0.04, num_leaves=31,
                    min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    verbose=-1)


def baseline(rows):
    """Persistence forecast: same hour on the last observed day (current AQI if missing)."""
    return rows["persist"].fillna(rows["aqi_now"]).to_numpy()


def bucket(h):
    return np.where(h <= 24, "24h", np.where(h <= 48, "48h", "72h"))


def scores(y, pred, h, model):
    d = pd.DataFrame({"err": pred - y, "b": bucket(h)}).dropna()
    return [{"model": model, "horizon": b, "mae": round(g["err"].abs().mean(), 1),
             "rmse": round(float(np.sqrt((g["err"] ** 2).mean())), 1)} for b, g in d.groupby("b")]


def prophet_scores(df, split, test):
    try:
        from prophet import Prophet
    except ImportError:
        print("  (Prophet not installed: skipping that baseline. pip install prophet to include it.)")
        return []
    city = df.groupby("ts")["aqi"].mean().dropna().reset_index().rename(columns={"ts": "ds", "aqi": "y"})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(city[city["ds"] < split])
    fc = m.predict(pd.DataFrame({"ds": pd.date_range(split, city["ds"].max(), freq="h")}))
    pred = fc.set_index("ds")["yhat"]
    # Prophet ignores recent observations, so its error is the same at every horizon.
    t = test.groupby("target_ts")["y"].mean()
    t = t[t.index.isin(pred.index)]
    rows = scores(t.to_numpy(), pred.reindex(t.index).to_numpy(), np.full(len(t), 24), "Prophet")
    return [dict(r, horizon=b) for r in rows for b in ("24h", "48h", "72h")]


def main():
    aqi_raw, wx = load_raw()
    df, wx = prepare(aqi_raw, wx)
    station_names = sorted(df["station"].unique())
    rows = make_rows(df, wx, station_names, origins="sampled", step=ORIGIN_STEP_HOURS).dropna(subset=["y"])
    split = rows["target_ts"].max() - pd.Timedelta(days=TEST_DAYS)
    train = rows[rows["target_ts"] < split]
    test = rows[rows["origin"] >= split]
    print(f"Training rows: {len(train):,}  Test rows: {len(test):,}  Split at {split}")

    # Predict the correction to the persistence forecast. Trees cannot extrapolate past
    # values seen in training, but corrections stay in a stable range as the season shifts.
    Xtr, Xte, yte = train[FEATURES], test[FEATURES], test["y"]
    ytr = train["y"] - baseline(train)
    base_te = baseline(test)
    cat = ["station_id"]

    # Early stopping on the last week of the training period (time-ordered, no leakage)
    val_cut = train["target_ts"].max() - pd.Timedelta(days=7)
    fit_mask = (train["target_ts"] < val_cut - pd.Timedelta(hours=72)).to_numpy()
    val_mask = (train["origin"] >= val_cut).to_numpy()
    probe = lgb.LGBMRegressor(**dict(POINT_PARAMS, n_estimators=2000)).fit(
        Xtr[fit_mask], ytr[fit_mask], eval_set=[(Xtr[val_mask], ytr[val_mask])],
        categorical_feature=cat, callbacks=[lgb.early_stopping(100, verbose=False)])
    best_n = max(50, probe.best_iteration_ or POINT_PARAMS["n_estimators"])
    print(f"  Early stopping chose {best_n} trees")
    point = lgb.LGBMRegressor(**dict(POINT_PARAMS, n_estimators=best_n)).fit(Xtr, ytr, categorical_feature=cat)
    q_params = dict(POINT_PARAMS, n_estimators=best_n)
    q10 = lgb.LGBMRegressor(**dict(q_params, objective="quantile", alpha=0.08)).fit(Xtr, ytr, categorical_feature=cat)
    q90 = lgb.LGBMRegressor(**dict(q_params, objective="quantile", alpha=0.92)).fit(Xtr, ytr, categorical_feature=cat)

    metrics = []
    metrics += scores(yte.to_numpy(), test["persist"].to_numpy(), test["horizon"].to_numpy(), "Persistence baseline")
    metrics += prophet_scores(df, split, test)
    metrics += scores(yte.to_numpy(), base_te + point.predict(Xte), test["horizon"].to_numpy(), "LightGBM + weather")
    lo, hi = base_te + q10.predict(Xte), base_te + q90.predict(Xte)
    coverage = float(((yte >= lo) & (yte <= hi)).mean())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    point.booster_.save_model(MODEL_DIR / "lgbm_point.txt")
    q10.booster_.save_model(MODEL_DIR / "lgbm_q10.txt")
    q90.booster_.save_model(MODEL_DIR / "lgbm_q90.txt")
    pd.DataFrame(metrics).to_csv(MODEL_DIR / "metrics.csv", index=False)
    (MODEL_DIR / "train_info.json").write_text(json.dumps({
        "split": str(split), "train_rows": len(train), "test_rows": len(test),
        "interval_coverage_10_90": round(coverage, 3), "features": FEATURES,
        "stations": station_names,
    }, indent=2))

    # SHAP: mean |contribution| per feature, in AQI points
    import shap
    sample = Xte.sample(min(3000, len(Xte)), random_state=0)
    sv = shap.TreeExplainer(point).shap_values(sample)
    imp = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES],
                        "mean_abs_shap": np.abs(sv).mean(axis=0).round(1)})
    imp.sort_values("mean_abs_shap", ascending=False).head(10).to_csv(MODEL_DIR / "shap.csv", index=False)

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"10-90% interval coverage on test: {coverage:.0%} (ideal ~80%)")


if __name__ == "__main__":
    main()