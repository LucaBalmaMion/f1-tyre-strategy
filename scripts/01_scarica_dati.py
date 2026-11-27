import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # per poter scrivere "from src..."

from src.data import DATA_DIR, GARE_CANDIDATE, carica_gara, nomi_circuiti


def main():
    circuito = nomi_circuiti()
    scaricate = []
    for session_key in GARE_CANDIDATE:
        dati = carica_gara(session_key)
        if dati is None:
            continue
        nome = circuito.get(session_key, session_key)
        print(f"{nome} ({session_key}): {len(dati['laps'])} giri, {len(dati['stints'])} stint, "
              f"{len(dati['weather'])} misure meteo, {len(dati['rc'])} messaggi di race control")
        scaricate.append(session_key)
    print(f"\nGare asciutte disponibili: {len(scaricate)} su {len(GARE_CANDIDATE)}. File in {DATA_DIR}")


if __name__ == "__main__":
    main()
