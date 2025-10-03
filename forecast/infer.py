#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, numpy as np, pandas as pd, pytz, joblib

TZ = os.getenv("TZ", "America/Santiago")
DEFAULT_INTERVAL_MIN = int(os.getenv("INTERVAL_MIN", "60"))
MODELS_DIR = os.getenv("MODELS_DIR", "models")
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_CALLS = os.path.join(MODELS_DIR, "rf_calls.pkl")
MODEL_TMO   = os.path.join(MODELS_DIR, "rf_tmo.pkl")

try:
    import holidays
    CL_HOLIDAYS = holidays.CL()
except Exception:
    CL_HOLIDAYS = None

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True).dt.tz_convert(TZ)
    d["year"] = d["datetime"].dt.year
    d["month"] = d["datetime"].dt.month
    d["day"] = d["datetime"].dt.day
    d["hour"] = d["datetime"].dt.hour
    d["minute"] = d["datetime"].dt.minute
    d["dow"] = d["datetime"].dt.dayofweek
    d["is_weekend"] = d["dow"].isin([5,6]).astype(int)
    if CL_HOLIDAYS is not None:
        d["date"] = d["datetime"].dt.date
        d["is_holiday"] = d["date"].apply(lambda x: int(x in CL_HOLIDAYS))
    else:
        d["is_holiday"] = 0
    d["day_of_year"] = d["datetime"].dt.dayofyear
    d["week_of_year"] = d["datetime"].dt.isocalendar().week.astype(int)
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24)
    d["sin_doy"] = np.sin(2*np.pi*d["day_of_year"]/366)
    d["cos_doy"] = np.cos(2*np.pi*d["day_of_year"]/366)
    def part_of_day(h):
        if h < 6: return "madrugada"
        if h < 12: return "manana"
        if h < 18: return "tarde"
        return "noche"
    d["part_of_day"] = d["hour"].apply(part_of_day)
    d["is_business_hour"] = ((d["hour"] >= 9) & (d["hour"] < 19)).astype(int)
    return d

def build_future_index(start, end, interval_min):
    tz = pytz.timezone(TZ)
    start = tz.localize(pd.Timestamp(start)).tz_convert("UTC")
    end = tz.localize(pd.Timestamp(end)).tz_convert("UTC")
    rng = pd.date_range(start=start, end=end, freq=f"{interval_min}min", inclusive="left", tz="UTC")
    return pd.DataFrame({"datetime": rng})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--interval_min", type=int, default=DEFAULT_INTERVAL_MIN)
    args = ap.parse_args()

    m_calls = joblib.load(MODEL_CALLS)
    m_tmo   = joblib.load(MODEL_TMO)

    future = build_future_index(args.start, args.end, args.interval_min)
    X = add_time_features(future)

    feat_calls = [c for c in m_calls["features"] if c in X.columns]
    feat_tmo   = [c for c in m_tmo["features"]   if c in X.columns]

    pred_calls = m_calls["pipeline"].predict(X[feat_calls])
    pred_tmo   = m_tmo["pipeline"].predict(X[feat_tmo])

    df = future.copy()
    df["fecha"] = X["datetime"].dt.strftime("%Y-%m-%d")
    df["hora"] = X["datetime"].dt.strftime("%H:%M")
    df["llamadas"] = np.maximum(pred_calls, 0).round(2)
    df["tmo"] = np.maximum(pred_tmo, 1).round(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "forecast.json")
    df[["fecha","hora","llamadas","tmo"]].to_json(out_path, orient="records", force_ascii=False)
    print("OK ->", out_path)

if __name__ == "__main__":
    main()
