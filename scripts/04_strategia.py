import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # per poter scrivere "from src..."

import numpy as np
import pandas as pd

from src.clean import numero_giri, segna_soste
from src.data import DATA_DIR, carica_gara, carica_puliti, nomi_circuiti
from src.models import allena_gara, mae_giri_puliti
from src.simulate import (backtest, controlli_buon_senso, descrivi, giri_bandiera_rossa, giri_neutralizzati,
                          migliore_strategia, stima_perdita_pit, tempo_gara)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    circuito = nomi_circuiti()

    riepilogo = []
    for session_key, puliti in carica_puliti().items():
        nome = circuito.get(session_key, session_key)
        g = carica_gara(session_key)          # tabelle grezze (dalla cache in data/)
        laps = segna_soste(g["laps"])
        n_giri = numero_giri(laps)
        modello, mescole, _, eta_max = allena_gara(puliti)
        pit = stima_perdita_pit(laps, puliti, giri_neutralizzati(laps, g["rc"]))

        print(f"\n================ {nome} ({session_key}), {n_giri} giri ================")
        print(f"Perdita per sosta: {pit['perdita']:.1f} s  (soste usate {pit['soste_usate']}, "
              f"scartate perché sporche/incomplete {pit['soste_scartate']})")

        if len(mescole) < 2:
            print("Meno di due mescole stimabili: impossibile rispettare la regola delle due mescole.")
            continue

        una = migliore_strategia(modello, mescole, n_giri, pit["perdita"], eta_max, 1)
        due = migliore_strategia(modello, mescole, n_giri, pit["perdita"], eta_max, 2)
        for etichetta, res in (("1 sosta ", una), ("2 soste", due)):
            if res is None:
                print(f"{etichetta}: nessun piano valido (stint troppo lunghi per l'età massima osservata)")
            else:
                # verifica: il tempo dalla cache coincide con quello di tempo_gara
                assert np.isclose(res["tempo_totale"], tempo_gara(modello, res["piano"], n_giri, pit["perdita"]))
                print(f"{etichetta}: {descrivi(res['piano']):40s} {res['tempo_totale']:9.1f} s")
        candidati = [r for r in (una, due) if r is not None]
        if not candidati:
            continue
        ottimo = min(candidati, key=lambda r: r["tempo_totale"])
        print(f"Strategia ottima: {descrivi(ottimo['piano'])}")

        print("Controlli di buon senso:")
        for nome_c, ok, dettaglio in controlli_buon_senso(modello, mescole, n_giri):
            print(f"  [{'OK  ' if ok else 'FAIL'}] {nome_c}: {dettaglio}")

        bt = backtest(laps, g["stints"], g["rc"], modello, n_giri, pit["perdita"], ottimo)
        if bt.empty:
            print("Back-testing: nessun pilota con strategia simulabile.")
            continue
        bt.to_csv(DATA_DIR / f"backtest_{session_key}.csv", index=False)
        print("Back-testing (primi 10 arrivati con strategia simulabile):")
        print(bt.head(10)[["pos", "pilota", "piano", "tempo_sim", "gap_ottimo", "reale_dal_giro_2", "errore_sim"]]
              .round(1).to_string(index=False))
        mae_giri = mae_giri_puliti(modello, puliti)
        vincitore = bt.iloc[0]
        spread = bt["tempo_sim"].quantile(0.9) - bt["tempo_sim"].quantile(0.1)
        err_gara = bt["errore_sim"].abs().median() if bt["errore_sim"].notna().any() else np.nan
        print(f"Errore del modello sui giri puliti: {mae_giri:.2f} s/giro")
        if giri_bandiera_rossa(g["rc"]):
            print("Gara con bandiera rossa: il tempo reale include la sospensione, confronto con la realtà non significativo")
        else:
            print(f"Errore simulato vs reale sull'intera gara (mediana): {err_gara:.1f} s")
        print(f"Differenza tra strategie reali (10°-90° percentile del tempo simulato): {spread:.1f} s")
        print(f"Il primo arrivato ({vincitore['pilota']}) è a {vincitore['gap_ottimo']:+.1f} s dall'ottimo simulato")
        riepilogo.append({"gara": nome, "perdita_s": round(pit["perdita"], 1), "ottimo": descrivi(ottimo["piano"]),
                          "tempo_ottimo_s": round(ottimo["tempo_totale"], 1),
                          "vincitore_gap_s": round(vincitore["gap_ottimo"], 1),
                          "mae_giri_puliti_s": round(mae_giri, 2), "errore_gara_s": round(err_gara, 1),
                          "spread_strategie_s": round(spread, 1)})

    riepilogo = pd.DataFrame(riepilogo)
    riepilogo.to_csv(DATA_DIR / "riepilogo_strategie.csv", index=False)
    print("\n================ Riepilogo strategie ================")
    print(riepilogo.to_string(index=False))


if __name__ == "__main__":
    main()
