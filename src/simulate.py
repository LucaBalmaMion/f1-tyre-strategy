import itertools

import numpy as np
import pandas as pd

from src.clean import PCT_TRANSITION, numero_giri

STINT_MIN = 5


# =============================================================================
# Perdita per sosta
# =============================================================================

def giri_bandiera_rossa(rc):
    return set(rc.loc[rc["flag"] == "RED", "lap_number"].dropna().astype(int))


def giri_neutralizzati(laps, rc):
    """Giri in cui una sosta è 'sporca': gruppo lento (SC, VSC, bandiera rossa) o bandiere in pista."""
    normali = laps[~laps["is_pit_out_lap"] & ~laps["is_pit_in_lap"] & (laps["lap_number"] > 1)
                   & laps["lap_duration"].notna()]
    mediane = normali.groupby("lap_number")["lap_duration"].median()
    riferimento = mediane.quantile(0.25)
    senza_tempi = set(range(2, numero_giri(laps) + 1)) - set(mediane.index)
    lenti = set(mediane[mediane > riferimento * (1 + PCT_TRANSITION)].index) | senza_tempi
    bandiere = rc[((rc["category"] == "Flag") & rc["flag"].isin(["YELLOW", "DOUBLE YELLOW", "RED"]))
                  | (rc["category"] == "SafetyCar")]
    return lenti | set(bandiere["lap_number"].dropna().astype(int))


def stima_perdita_pit(laps, giri_puliti, neutralizzati, n_rif=5, min_rif=3):
    """Mediana di (rientro - rif) + (uscita - rif), con rif = mediana degli ultimi giri puliti prima della sosta."""
    df = laps.sort_values(["driver_number", "lap_number"]).copy()
    df["tempo_uscita"] = df.groupby("driver_number")["lap_duration"].shift(-1)
    df["giro_uscita"] = df.groupby("driver_number")["lap_number"].shift(-1)
    soste = df[df["is_pit_in_lap"]]
    perdite, scartate = [], 0
    for _, r in soste.iterrows():
        if r["lap_number"] in neutralizzati or r["giro_uscita"] in neutralizzati:
            scartate += 1
            continue
        precedenti = giri_puliti[(giri_puliti["driver_number"] == r["driver_number"])
                                 & (giri_puliti["lap_number"] < r["lap_number"])].sort_values("lap_number").tail(n_rif)
        if len(precedenti) < min_rif or pd.isna(r["lap_duration"]) or pd.isna(r["tempo_uscita"]):
            scartate += 1
            continue
        rif = precedenti["lap_duration"].median()
        perdite.append((r["lap_duration"] - rif) + (r["tempo_uscita"] - rif))
    return {"perdita": float(np.median(perdite)) if perdite else np.nan,
            "soste_usate": len(perdite), "soste_scartate": scartate}


# =============================================================================
# Simulatore
# =============================================================================

def tempo_stint(modello, mescola, primo_giro, ultimo_giro, eta_iniziale=0):
    giri = np.arange(primo_giro, ultimo_giro + 1)
    return float(modello.predict(mescola, eta_iniziale + giri - primo_giro, giri).sum())


def tempo_gara(modello, piano, n_giri, perdita_sosta, dal_giro=1):
    """piano = lista di (mescola, ultimo_giro) o (mescola, ultimo_giro, età_iniziale).
    perdita_sosta = un numero, oppure una lista con la perdita di ogni sosta.
    L'ultimo stint deve finire a n_giri. dal_giro > 1 esclude i primi giri (per il confronto con la realtà)."""
    assert piano[-1][1] == n_giri, "l'ultimo stint deve finire all'ultimo giro"
    perdite = perdita_sosta if isinstance(perdita_sosta, list) else [perdita_sosta] * (len(piano) - 1)
    totale, primo = 0.0, 1
    for i, stint in enumerate(piano):
        mescola, fine = stint[0], stint[1]
        eta0 = stint[2] if len(stint) > 2 else 0
        inizio = max(primo, dal_giro)
        if inizio <= fine:
            totale += tempo_stint(modello, mescola, inizio, fine, eta0 + inizio - primo)
        if i < len(piano) - 1:
            totale += perdite[i]
        primo = fine + 1
    return totale


def migliore_strategia(modello, mescole, n_giri, perdita_sosta, eta_max, n_soste, stint_min=STINT_MIN):
    """Forza bruta su giri di sosta e mescole. eta_max = {} toglie il limite di età."""
    costo = {}

    def costo_stint(m, a, b):
        if (m, a, b) not in costo:
            costo[(m, a, b)] = tempo_stint(modello, m, a, b)
        return costo[(m, a, b)]

    migliore = None
    for soste in itertools.combinations(range(stint_min, n_giri - stint_min + 1), n_soste):
        confini = [0, *soste, n_giri]
        stint = [(confini[k] + 1, confini[k + 1]) for k in range(n_soste + 1)]
        if any(b - a + 1 < stint_min for a, b in stint):
            continue
        for mescole_piano in itertools.product(mescole, repeat=n_soste + 1):
            if len(set(mescole_piano)) < 2:                       # regola delle due mescole
                continue
            # età nell'ultimo giro dello stint = durata - 1: non oltre la massima vista nei dati
            if any(b - a > eta_max.get(m, np.inf) for m, (a, b) in zip(mescole_piano, stint)):
                continue
            t = sum(costo_stint(m, a, b) for m, (a, b) in zip(mescole_piano, stint)) + n_soste * perdita_sosta
            if migliore is None or t < migliore["tempo_totale"]:
                migliore = {"piano": [(m, b) for m, (a, b) in zip(mescole_piano, stint)],
                            "soste": list(soste), "tempo_totale": t}
    return migliore


def descrivi(piano):
    return " > ".join(f"{p[0]}({p[1]})" for p in piano[:-1]) + f" > {piano[-1][0]}"


# =============================================================================
# Controlli di buon senso
# =============================================================================

def controlli_buon_senso(modello, mescole, n_giri):
    """Le 4 proprietà della guida. Senza limite di età, per isolare la logica del simulatore."""
    risultati = []

    def best(mod, soste, perdita):
        return migliore_strategia(mod, mescole, n_giri, perdita, {}, soste)

    def esito(nome, ok, dettaglio):
        risultati.append((nome, ok, dettaglio))

    # 1. La perdita non cambia il piano ottimo, sposta il tempo della stessa quantità
    r = {p: best(modello, 1, p) for p in (15, 20, 25)}
    stesso_piano = len({tuple(x["piano"]) for x in r.values()}) == 1
    delta_ok = all(np.isclose(r[p]["tempo_totale"] - r[15]["tempo_totale"], p - 15) for p in r)
    esito("1. perdita non cambia il giro di sosta", stesso_piano and delta_ok,
          f"sosta al giro {r[15]['soste']} con 15/20/25 s")

    # 2. La perdita cambia il confronto 1 vs 2 soste (di esattamente Δperdita)
    diff = {p: best(modello, 2, p)["tempo_totale"] - best(modello, 1, p)["tempo_totale"] for p in (10, 20, 30)}
    esito("2. perdita cambia 2 soste - 1 sosta", np.isclose(diff[20] - diff[10], 10) and np.isclose(diff[30] - diff[20], 10),
          ", ".join(f"{p}s: {d:+.1f}" for p, d in diff.items()))

    # 3. Degrado zero -> la seconda sosta costa esattamente una perdita in più
    piatto = modello.con_degrado({m: 0.0 for m in mescole})
    d = best(piatto, 2, 20)["tempo_totale"] - best(piatto, 1, 20)["tempo_totale"]
    esito("3. degrado zero -> 2 soste = 1 sosta + perdita", np.isclose(d, 20), f"differenza {d:.2f} s")

    # 4. Raddoppiare il degrado: se la mescola è usata il tempo peggiora, se non lo è nulla cambia
    base = best(modello, 1, 20)
    usate = {m for m, _ in base["piano"]}
    for m in mescole:
        nuovo = best(modello.con_degrado({m: 2 * modello.degrado[m]}), 1, 20)
        dt = nuovo["tempo_totale"] - base["tempo_totale"]
        if m in usate:
            ok = dt > 0 if modello.degrado[m] > 0 else dt <= 0
            esito(f"4. degrado {m} x2 (usata)", ok, f"{descrivi(base['piano'])} -> {descrivi(nuovo['piano'])}, {dt:+.1f} s")
        else:
            ok = nuovo["piano"] == base["piano"] and np.isclose(dt, 0)
            esito(f"4. degrado {m} x2 (non usata)", ok, f"piano invariato: {nuovo['piano'] == base['piano']}")
    return risultati


# =============================================================================
# Back-testing
# =============================================================================

def piano_reale(s, n_giri):
    """Stint reali di un pilota -> piano [(mescola, ultimo_giro, età_iniziale)].
    Due stint consecutivi con la stessa gomma ed età continua sono lo stesso treno (OpenF1 a volte
    spezza uno stint, per esempio alla ripartenza dopo una bandiera rossa): li unisco."""
    piano = []
    for c, inizio, fine, eta0 in zip(s["compound"], s["lap_start"], s["lap_end"], s["tyre_age_at_start"].fillna(0)):
        if piano and piano[-1][0] == c and eta0 == piano[-1][2] + (piano[-1][1] - piano[-1][3]) + 1:
            piano[-1] = (c, fine, piano[-1][2], piano[-1][3])
        else:
            piano.append((c, fine, eta0, inizio))
    piano = [(c, int(fine), int(eta0)) for c, fine, eta0, _ in piano]
    piano[-1] = (piano[-1][0], n_giri, piano[-1][2])
    return piano


def backtest(laps, stints, rc, modello, n_giri, perdita_sosta, ottimo):
    """Simula le strategie reali dei piloti arrivati al traguardo e le confronta con l'ottimo e con la realtà."""
    rossa = giri_bandiera_rossa(rc)
    ultimo = laps[laps["lap_number"] == n_giri].sort_values("date_start")
    ordine_arrivo = list(ultimo["driver_number"])
    righe = []
    for pos, pilota in enumerate(ordine_arrivo, start=1):
        s = stints[stints["driver_number"] == pilota].sort_values("stint_number")
        if s.empty or not s["compound"].isin(modello.mescole).all() or s["lap_end"].iloc[:-1].isna().any():
            continue
        piano = piano_reale(s, n_giri)
        # Un cambio gomme durante la bandiera rossa non costa tempo ai box
        perdite = [0.0 if (fine in rossa or fine + 1 in rossa) else perdita_sosta for _, fine, _ in piano[:-1]]
        # Tempo reale dall'inizio del giro 2 al traguardo, dai timestamp (robusto ai lap_duration mancanti)
        giri_pilota = laps[laps["driver_number"] == pilota].set_index("lap_number")
        reale = np.nan
        # (la riga dopo l'ultimo giro, se c'è, è il rientro ai box dopo la bandiera a scacchi: non va usata)
        if 2 in giri_pilota.index and n_giri in giri_pilota.index and pd.notna(giri_pilota.loc[n_giri, "lap_duration"]):
            ultimo_giro = giri_pilota.loc[n_giri]
            arrivo = ultimo_giro["date_start"] + pd.to_timedelta(ultimo_giro["lap_duration"], unit="s")
            reale = (arrivo - giri_pilota.loc[2, "date_start"]).total_seconds()
        righe.append({
            "pos": pos, "pilota": pilota, "piano": descrivi(piano), "soste": len(piano) - 1,
            "tempo_sim": tempo_gara(modello, piano, n_giri, perdite),
            "sim_dal_giro_2": tempo_gara(modello, piano, n_giri, perdite, dal_giro=2),
            "reale_dal_giro_2": reale if not rossa else np.nan,   # con bandiera rossa il tempo reale include la sospensione
        })
    bt = pd.DataFrame(righe)
    if bt.empty:
        return bt
    bt["gap_ottimo"] = bt["tempo_sim"] - ottimo["tempo_totale"]
    bt["errore_sim"] = bt["sim_dal_giro_2"] - bt["reale_dal_giro_2"]
    return bt
