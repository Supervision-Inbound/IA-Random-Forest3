#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

warnings.filterwarnings("ignore")

TZ = os.getenv("TZ", "America/Santiago")
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
N_TREES = int(os.getenv("N_TREES", "800"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

TRAFFIC_XLSX = os.getenv("TRAFFIC_XLSX", "data/Historico trafico - tmo.xlsx")
HOLIDAYS_XLSX = os.getenv("HOLIDAYS_XLSX", "data/Feriados_Chile_2023_2027.xlsx")
TRAFFIC_SHEET = os.getenv("TRAFFIC_SHEET", "Sheet1")
HOLIDAYS_SHEET = os.getenv("HOLIDAYS_SHEET", "Sheet1")

OUT_DIR = os.getenv("OUT_DIR", "models")
os.makedirs(OUT_DIR, exist_ok=True)

def read_traffic(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "datetime": colmap[c] = "datetime"
        elif "suma de recibido" in lc or "recibido" in lc: colmap[c] = "calls"
        elif lc == "tmo": colmap[c] = "tmo_sec"
        elif "abandonado" in lc: colmap[c] = "abandonados"
        elif "contestado" in lc or "contestados" in lc: colmap[c] = "contestados"
    df = df.rename(columns=colmap)
    assert {"datetime","calls","tmo_sec"}.issubset(df.columns), f"Faltan columnas: {df.columns}"

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df["datetime"] = df["datetime"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")

    agg = {"calls":"sum","tmo_sec":"mean"}
    if "abandonados" in df.columns: agg["abandonados"]="sum"
    if "contestados" in df.columns: agg["contestados"]="sum"
    df = df.groupby("datetime", as_index=False).agg(agg)
    return df

def read_holidays(path, sheet):
    h = pd.read_excel(path, sheet_name=sheet)
    h.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in h.columns]
    if "Fecha" in h.columns: h = h.rename(columns={"Fecha":"date"})
    elif "fecha" in h.columns: h = h.rename(columns={"fecha":"date"})
    else: h = h.rename(columns={h.columns[0]:"date"})
    h["date"] = pd.to_datetime(h["date"]).dt.date
    return set(h["date"].tolist())

def add_time_features(df, holidays_set=None):
    d = df.copy()
    dt = d["datetime"].dt.tz_convert(TZ)
    d["year"]=dt.dt.year; d["month"]=dt.dt.month; d["day"]=dt.dt.day
    d["hour"]=dt.dt.hour; d["minute"]=dt.dt.minute
    d["dow"]=dt.dt.dayofweek; d["is_weekend"]=d["dow"].isin([5,6]).astype(int)
    d["date"]=dt.dt.date
    d["is_holiday"]=d["date"].apply(lambda x: int(x in holidays_set) if holidays_set else 0)
    d["day_of_year"]=dt.dt.dayofyear; d["week_of_year"]=dt.dt.isocalendar().week.astype(int)
    d["sin_hour"]=np.sin(2*np.pi*d["hour"]/24); d["cos_hour"]=np.cos(2*np.pi*d["hour"]/24)
    d["sin_doy"]=np.sin(2*np.pi*d["day_of_year"]/366); d["cos_doy"]=np.cos(2*np.pi*d["day_of_year"]/366)
    def part_of_day(h):
        if h < 6: return "madrugada"
        if h < 12: return "manana"
        if h < 18: return "tarde"
        return "noche"
    d["part_of_day"]=d["hour"].apply(part_of_day)
    d["is_business_hour"]=((d["hour"]>=9)&(d["hour"]<19)).astype(int)
    return d

def add_lags_rolls(df, col, lags=(1,24,168), rolls=(24,168)):
    d = df.sort_values("datetime").copy()
    for l in lags: d[f"{col}_lag_{l}"]=d[col].shift(l)
    for r in rolls: d[f"{col}_rollmean_{r}"]=d[col].shift(1).rolling(r, min_periods=max(1,int(r/4))).mean()
    return d

def ts_train_eval(df, target, feature_cols, name):
    use = df.dropna(subset=feature_cols+[target]).copy()
    X = use[feature_cols]; y = use[target]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num","passthrough",num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    rf = RandomForestRegressor(
        n_estimators=N_TREES,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    pipe = Pipeline([("pre", pre), ("rf", rf)])

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    maes=[]; rmses=[]; mapes=[]
    for tr, va in tscv.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict(X.iloc[va])
        mae = mean_absolute_error(y.iloc[va], p)
        rmse = mean_squared_error(y.iloc[va], p, squared=False)
        mape = (np.abs((y.iloc[va]-p)/np.clip(np.abs(y.iloc[va]),1e-6,None))).mean()
        maes.append(mae); rmses.append(rmse); mapes.append(mape)

    print(f"{name} | MAE={np.mean(maes):.3f} RMSE={np.mean(rmses):.3f} MAPE={np.mean(mapes)*100:.2f}% (n={len(use)})")

    pipe.fit(X, y)
    out = os.path.join(OUT_DIR, f"{name}.pkl")
    joblib.dump({"pipeline": pipe, "features": feature_cols}, out)
    print("Guardado:", out)

def main():
    df = read_traffic(TRAFFIC_XLSX, TRAFFIC_SHEET)
    holi = read_holidays(HOLIDAYS_XLSX, HOLIDAYS_SHEET)

    fe = add_time_features(df, holidays_set=holi)
    fe = add_lags_rolls(fe, "calls", lags=(1,24,168), rolls=(24,168))
    fe = add_lags_rolls(fe, "tmo_sec", lags=(1,24,168), rolls=(24,168))

    features = [
        "year","month","day","hour","minute","dow","is_weekend","is_holiday",
        "day_of_year","week_of_year","sin_hour","cos_hour","sin_doy","cos_doy",
        "part_of_day","is_business_hour",
        "calls_lag_1","calls_lag_24","calls_lag_168","calls_rollmean_24","calls_rollmean_168",
        "tmo_sec_lag_1","tmo_sec_lag_24","tmo_sec_lag_168","tmo_sec_rollmean_24","tmo_sec_rollmean_168",
    ]

    ts_train_eval(fe, "calls", features, "rf_calls")
    ts_train_eval(fe, "tmo_sec", features, "rf_tmo")

if __name__ == "__main__":
    main()
