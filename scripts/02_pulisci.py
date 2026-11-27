"""Passo 2: pulisce i giri di ogni gara e salva in data/ i giri puliti e la tabella degli scarti."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # per poter scrivere "from src..."

import pandas as pd

from src.clean import prepara_gara
from src.data import DATA_DIR, GARE_CANDIDATE, carica_gara, nomi_circuiti, percorso_puliti


def main():
    circuito = nomi_circuiti()
    tabelle = []
    for session_key in GARE_CANDIDATE:
        dati = carica_gara(session_key)
        if dati is None:
            percorso_puliti(session_key).unlink(missing_ok=True)   # niente file vecchi di gare ora saltate
            continue
        puliti, funnel = prepara_gara(dati["laps"], dati["stints"], dati["weather"], dati["rc"])
        puliti["session_key"] = session_key
        print(f"\n--- {circuito.get(session_key, session_key)} ({session_key}): tabella degli scarti")
        print(funnel.to_string(index=False))
        puliti.to_csv(percorso_puliti(session_key), index=False)
        funnel.to_csv(DATA_DIR / f"scarti_{session_key}.csv", index=False)
        tabelle.append(puliti)

    if not tabelle:
        raise SystemExit("Nessuna gara disponibile: lancia prima scripts/01_scarica_dati.py")
    tutti = pd.concat(tabelle, ignore_index=True)
    assert len(tutti) == sum(len(t) for t in tabelle)
    assert not tutti[["compound", "tyre_age", "lap_number", "lap_duration"]].isna().any().any(), "NaN nelle feature"
    print(f"\nGiri puliti in totale: {len(tutti)} in {len(tabelle)} gare")


if __name__ == "__main__":
    main()
