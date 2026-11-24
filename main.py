import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# CONFIGURAZIONE
# ==================================================

BASE_URL = "https://api.openf1.org/v1"
DATA_DIR = "data"

SESSION_KEY = 9094
YEAR = 2023
SESSION_NAME = "Race"

os.makedirs(DATA_DIR, exist_ok=True)


# ==================================================
# DOWNLOAD E CARICAMENTO DATI
# ==================================================

def scarica(endpoint, params, nome_file):
    """
    Scarica i dati da OpenF1 oppure carica il CSV
    già presente nella cartella data.
    """

    percorso = os.path.join(DATA_DIR, nome_file)

    if os.path.exists(percorso):
        return pd.read_csv(percorso)

    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    dati = response.json()
    df = pd.DataFrame(dati)

    # Rimuove le colonne che contengono liste
    colonne_lista = [
        col for col in df.columns
        if df[col].apply(lambda x: isinstance(x, list)).any()
    ]

    df = df.drop(columns=colonne_lista)

    df.to_csv(percorso, index=False)

    time.sleep(1)

    return df


# ==================================================
# ACQUISIZIONE DEI DATASET
# ==================================================

df_sessions = scarica(
    "sessions",
    {"year": YEAR, "session_name": SESSION_NAME},
    "sessions.csv"
)

df_laps = scarica(
    "laps",
    {"session_key": SESSION_KEY},
    "laps.csv"
)

df_stints = scarica(
    "stints",
    {"session_key": SESSION_KEY},
    "stints.csv"
)

df_race_control = scarica(
    "race_control",
    {"session_key": SESSION_KEY},
    "race_control.csv"
)

df_weather = scarica(
    "weather",
    {"session_key": SESSION_KEY},
    "weather.csv"
)


# ==================================================
# PREPARAZIONE DEI GIRI
# ==================================================

df_laps = df_laps.copy()

df_laps["date_start"] = pd.to_datetime(
    df_laps["date_start"],
    format="ISO8601",
    utc=True
)

df_laps = df_laps.dropna(
    subset=["lap_duration", "date_start"]
)

# Esclude i giri di uscita dai box
df_laps = df_laps[
    df_laps["is_pit_out_lap"] != True
]


# ==================================================
# ASSOCIAZIONE GIRI E STINT
# ==================================================

df_laps_stints = pd.merge(
    df_laps,
    df_stints,
    on=["driver_number", "meeting_key", "session_key"],
    how="inner"
)

# Mantiene solamente i giri appartenenti allo stint
df_laps_stints = df_laps_stints[
    (df_laps_stints["lap_number"] >= df_laps_stints["lap_start"]) &
    (df_laps_stints["lap_number"] <= df_laps_stints["lap_end"])
].copy()

# Calcola l'età della gomma
df_laps_stints["tyre_age"] = (
    df_laps_stints["tyre_age_at_start"] +
    df_laps_stints["lap_number"] -
    df_laps_stints["lap_start"]
)


# ==================================================
# ASSOCIAZIONE DATI METEO
# ==================================================

df_weather = df_weather.copy()

df_weather["date"] = pd.to_datetime(
    df_weather["date"],
    format="ISO8601",
    utc=True
)

df_laps_stints = df_laps_stints.sort_values("date_start")
df_weather = df_weather.sort_values("date")

df_laps_stints = pd.merge_asof(
    df_laps_stints,
    df_weather,
    left_on="date_start",
    right_on="date",
    direction="backward"
)


# ==================================================
# ESCLUSIONE GIRI CON BANDIERE GIALLE
# ==================================================

bandiere_gialle = df_race_control[
    (df_race_control["category"] == "Flag") &
    (df_race_control["flag"].isin(["YELLOW", "DOUBLE YELLOW"]))
]

giri_da_escludere = bandiere_gialle["lap_number"].dropna().unique()

df_laps_stints = df_laps_stints[
    ~df_laps_stints["lap_number"].isin(giri_da_escludere)
].copy()

# ==================================================
# CLASSIFICAZIONE DELLE CONDIZIONI METEO
# ==================================================

# Riepilogo delle condizioni per ogni giro
riepilogo_giri = (
    df_laps_stints
    .groupby("lap_number")
    .agg(
        tempo_mediano=("lap_duration", "median"),
        quota_pioggia=("rainfall", "mean")
    )
    .reset_index()
)

# Riferimento dei tempi in condizioni asciutte
giri_asciutti = riepilogo_giri[
    (riepilogo_giri["lap_number"] <= 49) &
    (riepilogo_giri["quota_pioggia"] == 0)
]

tempo_asciutto = giri_asciutti["tempo_mediano"].median()


# Classifica le condizioni della pista
def classifica_fase(riga):

    if riga["tempo_mediano"] >= tempo_asciutto + 6:
        return "bagnato"

    elif (
        riga["quota_pioggia"] > 0 or
        riga["tempo_mediano"] >= tempo_asciutto + 2
    ):
        return "transizione"

    else:
        return "asciutto"


riepilogo_giri["fase_meteo"] = riepilogo_giri.apply(
    classifica_fase,
    axis=1
)

# Identifica i cambiamenti di fase
riepilogo_giri = riepilogo_giri.sort_values("lap_number")

riepilogo_giri["fase_id"] = (
    riepilogo_giri["fase_meteo"]
    .ne(riepilogo_giri["fase_meteo"].shift())
    .cumsum()
)

# Riporta la fase meteo su ogni giro di ogni pilota
df_laps_stints = df_laps_stints.merge(
    riepilogo_giri[["lap_number", "fase_meteo", "fase_id"]],
    on="lap_number",
    how="left"
)


# ==================================================
# IDENTIFICAZIONE OUTLIER
# ==================================================

df_laps_stints = df_laps_stints.sort_values(
    ["driver_number", "stint_number", "lap_number"]
)

# Mediana mobile separata per pilota, stint e fase meteo
df_laps_stints["rolling_median"] = (
    df_laps_stints
    .groupby(
        ["driver_number", "stint_number", "fase_id"]
    )["lap_duration"]
    .transform(
        lambda serie: serie.rolling(
            window=7,
            center=True,
            min_periods=5
        ).median()
    )
)

# Scostamento rispetto alla mediana mobile
df_laps_stints["scostamento"] = (
    df_laps_stints["lap_duration"] -
    df_laps_stints["rolling_median"]
)

# Identifica i giri anormalmente lenti
df_laps_stints["outlier"] = (
    df_laps_stints["scostamento"] > 2.5
)


# ==================================================
# RISULTATI
# ==================================================

outliers = df_laps_stints[
    df_laps_stints["outlier"]
].copy()
print(outliers)