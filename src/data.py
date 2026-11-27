import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"

# Percorsi assoluti: gli script funzionano da qualsiasi cartella vengano lanciati
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Gare candidate (2023). Quelle con pioggia o non scaricabili vengono saltate con un messaggio.
# 7953 Sakhir, 9094 Monte Carlo, 9102 Catalunya, 9157 Monza, 9181 Mexico City
ANNO = 2023
GARE_CANDIDATE = [7953, 9094, 9102, 9157, 9181]
MAX_QUOTA_PIOGGIA = 0.0          # gara "asciutta" = nessuna misura meteo con pioggia


def scarica(endpoint, params, nome_file):
    """Scarica da OpenF1 oppure legge il CSV già salvato in data/."""
    percorso = DATA_DIR / nome_file
    if percorso.exists():
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


def carica_sessioni(anno=ANNO):
    """Elenco delle gare dell'anno (serve per i nomi dei circuiti)."""
    return scarica("sessions", {"year": anno, "session_name": "Race"}, "sessions.csv")


def nomi_circuiti():
    """{session_key: nome breve del circuito}"""
    return carica_sessioni().set_index("session_key")["circuit_short_name"].to_dict()


def carica_gara(session_key):
    """Scarica le 4 tabelle di una gara. Restituisce None (con un messaggio) se la gara va saltata."""
    try:
        weather = scarica("weather", {"session_key": session_key}, f"weather_{session_key}.csv")
        if weather.empty:
            print(f"Gara {session_key}: meteo assente, la salto")
            return None
        quota_pioggia = weather["rainfall"].mean()
        if quota_pioggia > MAX_QUOTA_PIOGGIA:
            print(f"Gara {session_key}: pioggia nel {quota_pioggia:.0%} delle misure, la salto (serve una gara asciutta)")
            return None
        laps = scarica("laps", {"session_key": session_key}, f"laps_{session_key}.csv")
        stints = scarica("stints", {"session_key": session_key}, f"stints_{session_key}.csv")
        rc = scarica("race_control", {"session_key": session_key}, f"rc_{session_key}.csv")
    except requests.RequestException as e:
        print(f"Gara {session_key}: download non riuscito ({e.__class__.__name__}: {str(e)[:80]}), la salto")
        return None
    if laps.empty or stints.empty:
        print(f"Gara {session_key}: giri o stint assenti, la salto")
        return None
    return {"laps": laps, "stints": stints, "weather": weather, "rc": rc}


# --- Risultati intermedi: lo script 02 li scrive, gli script 03 e 04 li leggono ---

def percorso_puliti(session_key):
    return DATA_DIR / f"giri_puliti_{session_key}.csv"


def carica_puliti():
    """{session_key: giri puliti} per le gare preparate dallo script 02."""
    gare = {}
    for session_key in GARE_CANDIDATE:
        percorso = percorso_puliti(session_key)
        if percorso.exists():
            gare[session_key] = pd.read_csv(percorso)
    if not gare:
        raise SystemExit("Nessun file giri_puliti_*.csv in data/: lancia prima scripts/02_pulisci.py")
    return gare
