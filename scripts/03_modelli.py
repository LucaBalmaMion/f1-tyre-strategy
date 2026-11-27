import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # per poter scrivere "from src..."

import pandas as pd

from src.data import DATA_DIR, FIG_DIR, carica_puliti, nomi_circuiti
from src.models import allena_gara, grafico_degrado, validazione_una_gara_fuori


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    circuito = nomi_circuiti()
    gare = carica_puliti()

    # --- Degrado per gara e mescola ---------------------------------------------
    righe_degrado = []
    for session_key, puliti in gare.items():
        nome = circuito.get(session_key, session_key)
        modello, mescole, conteggi, eta_max = allena_gara(puliti)
        escluse = [m for m in conteggi.index if m not in mescole]

        print(f"\n================ {nome} ({session_key}) ================")
        if escluse:
            print(f"Mescole con troppi pochi dati, escluse: {escluse}")
        print(f"Carburante/evoluzione pista: {modello.carburante:+.3f} s per giro")
        for m in mescole:
            c = conteggi.loc[m]
            print(f"  {m:6s} passo {modello.passo[m]:7.2f} s | degrado {modello.degrado[m]:+.3f} s/giro | "
                  f"età max {eta_max[m]:2d} | {c['giri']} giri, {c['piloti']} piloti")
            righe_degrado.append({"gara": nome, "session_key": session_key, "mescola": m,
                                  "degrado_s_giro": modello.degrado[m], "giri": c["giri"], "piloti": c["piloti"]})
            if modello.degrado[m] <= 0:
                print(f"  ATTENZIONE: degrado {m} non positivo, la stima non è credibile")

    tabella = pd.DataFrame(righe_degrado)
    tabella.to_csv(DATA_DIR / "degrado_per_gara.csv", index=False)
    grafico_degrado(tabella, FIG_DIR / "degrado_per_gara.png")
    print("\n================ Degrado per gara e mescola ================")
    print(tabella.round(3).to_string(index=False))

    # --- Validazione "una gara fuori" (sezione 11) ------------------------------
    tutti = pd.concat(gare.values(), ignore_index=True)
    validazione = validazione_una_gara_fuori(tutti)
    validazione.to_csv(DATA_DIR / "validazione_una_gara_fuori.csv", index=False)
    print("\n================ Validazione una gara fuori (MAE in s, tempi relativi alla mediana di gara) ================")
    print(validazione.to_string(index=False))


if __name__ == "__main__":
    main()
