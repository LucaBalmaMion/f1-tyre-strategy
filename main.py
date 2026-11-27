import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import itertools

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict



BASE_URL = "https://api.openf1.org/v1"
DATA_DIR = "data"

# Soglie della pulizia
PCT_TRANSITION, PCT_WET = 0.03, 0.08
MEDIAN_WINDOW, MIN_PERIODS, PCT_OUTLIER = 7, 5, 0.025
os.makedirs(DATA_DIR, exist_ok=True)


def scarica(endpoint, params, nome_file):
    """Scarica da OpenF1 oppure legge il CSV già salvato in data/."""
    percorso = os.path.join(DATA_DIR, nome_file)
    if os.path.exists(percorso):
        return pd.read_csv(percorso)
    response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    if df.empty:
        print(f"Attenzione: nessun dato per {endpoint} {params}, file non salvato")
        return df
    df = df.drop(columns=[c for c in df.columns if df[c].apply(lambda x: isinstance(x, list)).any()])
    df.to_csv(percorso, index=False)
    time.sleep(1)
    return df


funnel = []


def registra(regola, tabella):
    funnel.append((regola, len(tabella)))


def prepara_gara(laps, stints, weather, rc):
    funnel = []

    # Giri: giro di rientro = giro prima di un'uscita (calcolato prima di togliere righe)
    laps = laps.copy()
    laps["date_start"] = pd.to_datetime(laps["date_start"], format="ISO8601", utc=True).dt.as_unit("ns")
    laps["is_pit_out_lap"] = laps["is_pit_out_lap"].eq(True)
    laps = laps.sort_values(["driver_number", "lap_number"])
    laps["is_pit_in_lap"] = laps.groupby("driver_number")["is_pit_out_lap"].shift(-1, fill_value=False)
    registra("giri scaricati", laps)

    laps = laps.dropna(subset=["lap_duration", "date_start"])
    registra("tempo e data presenti", laps)

    laps = laps[~laps["is_pit_out_lap"]]
    registra("non giro di uscita", laps)

    laps = laps[~laps["is_pit_in_lap"]]
    registra("non giro di rientro", laps)

    laps = laps[laps["lap_number"] > 1]
    registra("non giro 1", laps)

    # Giri + stint: ogni giro prende lo stint che lo contiene, poi si calcola l'età della gomma
    laps_stints = laps.merge(stints, on=["driver_number", "meeting_key", "session_key"])
    laps_stints = laps_stints[
        laps_stints["lap_number"].between(laps_stints["lap_start"], laps_stints["lap_end"])].copy()
    registra("giri con uno stint", laps_stints)

    laps_stints["tyre_age"] = laps_stints["tyre_age_at_start"] + laps_stints["lap_number"] - laps_stints[
        "lap_start"]

    # Meteo: ultima misura precedente all'inizio del giro (senza le chiavi già presenti nei giri)
    weather = weather.drop(columns=["meeting_key", "session_key"])
    weather["date"] = pd.to_datetime(weather["date"], format="ISO8601", utc=True).dt.as_unit("ns")
    laps_stints = pd.merge_asof(laps_stints.sort_values("date_start"), weather.sort_values("date"),
                                left_on="date_start", right_on="date", direction="backward")

    # Bandiere gialle: scarta i giri interessati, per tutti i piloti
    gialle = rc[
        (rc["category"] == "Flag") & rc["flag"].isin(["YELLOW", "DOUBLE YELLOW"])]
    laps_stints = laps_stints[~laps_stints["lap_number"].isin(gialle["lap_number"].dropna().unique())].copy()
    registra("senza bandiere gialle", laps_stints)

    # Fase meteo di ogni giro (asciutto / transizione / bagnato) e sua numerazione
    riepilogo_giri = laps_stints.groupby("lap_number").agg(tempo_mediano=("lap_duration", "median"),
                                                           quota_pioggia=("rainfall", "mean")).reset_index()
    asciutti = riepilogo_giri[riepilogo_giri["quota_pioggia"] == 0]
    tempo_asciutto = asciutti["tempo_mediano"].quantile(0.25)

    def classifica_fase(riga):
        if riga["tempo_mediano"] >= tempo_asciutto * (1 + PCT_WET):
            return "bagnato"
        if riga["quota_pioggia"] > 0 or riga["tempo_mediano"] >= tempo_asciutto * (1 + PCT_TRANSITION):
            return "transizione"
        return "asciutto"

    riepilogo_giri["fase_meteo"] = riepilogo_giri.apply(classifica_fase, axis=1)
    riepilogo_giri["fase_id"] = riepilogo_giri["fase_meteo"].ne(riepilogo_giri["fase_meteo"].shift()).cumsum()
    laps_stints = laps_stints.merge(riepilogo_giri[["lap_number", "fase_meteo", "fase_id"]], on="lap_number",
                                    how="left")
    laps_stints = laps_stints[laps_stints["fase_meteo"] == "asciutto"].copy()
    registra("solo fase asciutta", laps_stints)

    # Outlier: giri più lenti di SOGLIA_OUTLIER secondi rispetto alla mediana mobile di pilota, stint e fase.
    # Dove la mediana non è calcolabile (primo/ultimo giro del gruppo) lo scostamento è NaN: il giro non viene segnalato.
    def mediana_mobile(serie):
        return serie.rolling(MEDIAN_WINDOW, center=True, min_periods=MIN_PERIODS).median()

    laps_stints = laps_stints.sort_values(["driver_number", "stint_number", "lap_number"])
    laps_stints["rolling_median"] = laps_stints.groupby(["driver_number", "stint_number", "fase_id"])[
        "lap_duration"].transform(mediana_mobile)
    laps_stints["scostamento"] = laps_stints["lap_duration"] - laps_stints["rolling_median"]
    laps_stints["outlier"] = laps_stints["scostamento"] > laps_stints['rolling_median'] * PCT_OUTLIER
    giri_puliti = laps_stints[~laps_stints["outlier"]].copy()
    registra("non outlier", giri_puliti)
    return giri_puliti


tabelle = []
for session_key in [7953, 9157, 9094, 9181]:
    laps = scarica('laps', {'session_key': session_key}, f'laps_{session_key}.csv')
    stints = scarica('stints', {'session_key': session_key}, f'stints_{session_key}.csv')
    weather = scarica('weather', {'session_key': session_key}, f'weather_{session_key}.csv')
    rc = scarica('race_control', {'session_key': session_key}, f'rc_{session_key}.csv')
    puliti = prepara_gara(laps, stints, weather, rc)
    puliti['session_key'] = session_key
    tabelle.append(puliti)
tutti = pd.concat(tabelle, ignore_index=True)

def stima_perdita_pit(laps, giri_puliti, n_rif=5, min_rif=3):
    df = laps.sort_values(["driver_number", "lap_number"]).copy()
    df["tempo_uscita"] = df.groupby("driver_number")["lap_duration"].shift(-1)
    soste = df[df["is_pit_in_lap"]]
    perdite = []
    for _, r in soste.iterrows():
        precedenti = giri_puliti[(giri_puliti["driver_number"] == r["driver_number"]) & (giri_puliti["lap_number"] < r["lap_number"])].tail(n_rif)
        if len(precedenti) < min_rif or pd.isna(r["lap_duration"]) or pd.isna(r["tempo_uscita"]):
            continue
        rif = precedenti["lap_duration"].median()
        perdite.append((r["lap_duration"] - rif) + (r["tempo_uscita"] - rif))
    return np.median(perdite) if perdite else np.nan

FEATURES = ["compound", "tyre_age", "lap_number", "track_temperature", "humidity"]

def nuovo_modello():
    pre = ColumnTransformer(transformers=[("comp", OneHotEncoder(handle_unknown="ignore"), ["compound"]),],remainder="passthrough",)
    return Pipeline([("pre", pre),("m", HistGradientBoostingRegressor(random_state=0)),])

modelli = {}

for session_key, giri_gara in tutti.groupby("session_key"):
    X = giri_gara[FEATURES]
    y = giri_gara["lap_duration"]
    modello = nuovo_modello()
    modello.fit(X, y)
    modelli[session_key] = modello


def meteo_per_giro(giri, n_giri):
    return (giri.groupby("lap_number")[["track_temperature", "humidity"]].median()
            .reindex(range(1, n_giri + 1)).interpolate().bfill().ffill())


def tempo_giro(model, meteo_gara, compound, tyre_age, lap_number):
    m = meteo_gara.loc[lap_number]
    riga = pd.DataFrame([{"compound": compound, "tyre_age": tyre_age, "lap_number": lap_number,
                          "track_temperature": m["track_temperature"], "humidity": m["humidity"]}])[FEATURES]
    return float(model.predict(riga)[0])




