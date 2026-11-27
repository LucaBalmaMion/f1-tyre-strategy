import os
import time
import itertools

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_URL = "https://api.openf1.org/v1"
DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Gare candidate (2023). Quelle con pioggia o non scaricabili vengono saltate con un messaggio.
# 7953 Sakhir, 9094 Monte Carlo, 9102 Catalunya, 9157 Monza, 9181 Mexico City
GARE_CANDIDATE = [7953, 9094, 9102, 9157, 9181]
MAX_QUOTA_PIOGGIA = 0.0          # gara "asciutta" = nessuna misura meteo con pioggia

# Soglie della pulizia
PCT_TRANSITION, PCT_WET = 0.03, 0.08
MEDIAN_WINDOW, MIN_PERIODS, PCT_OUTLIER = 7, 5, 0.025

# Modello e simulatore
MESCOLE_ASCIUTTE = ["SOFT", "MEDIUM", "HARD"]
MIN_GIRI_MESCOLA, MIN_PILOTI_MESCOLA = 50, 3   # sotto queste soglie la mescola non è stimabile
STINT_MIN = 5
COLORI_MESCOLA = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}


# =============================================================================
# Download
# =============================================================================

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


# =============================================================================
# Pulizia (sezioni 5 e 11)
# =============================================================================

def segna_soste(laps):
    """Converte le date e aggiunge is_pit_in_lap (= giro prima di un giro di uscita)."""
    laps = laps.copy()
    laps["date_start"] = pd.to_datetime(laps["date_start"], format="ISO8601", utc=True).dt.as_unit("ns")
    laps["is_pit_out_lap"] = laps["is_pit_out_lap"].eq(True)
    laps = laps.sort_values(["driver_number", "lap_number"])
    laps["is_pit_in_lap"] = laps.groupby("driver_number")["is_pit_out_lap"].shift(-1, fill_value=False)
    return laps


def prepara_gara(laps, stints, weather, rc):
    """Restituisce (giri puliti di una gara, tabella degli scarti)."""
    funnel = []

    def registra(regola, tabella):
        funnel.append((regola, len(tabella)))

    laps = segna_soste(laps)
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

    # Età = 0 nel primo giro dello stint (con gomma nuova). Il simulatore usa la stessa convenzione.
    laps_stints["tyre_age"] = laps_stints["tyre_age_at_start"] + laps_stints["lap_number"] - laps_stints["lap_start"]

    # Meteo: ultima misura precedente all'inizio del giro (senza le chiavi già presenti nei giri)
    weather = weather.drop(columns=["meeting_key", "session_key"])
    weather["date"] = pd.to_datetime(weather["date"], format="ISO8601", utc=True).dt.as_unit("ns")
    laps_stints = pd.merge_asof(laps_stints.sort_values("date_start"), weather.sort_values("date"),
                                left_on="date_start", right_on="date", direction="backward")

    # Bandiere gialle: scarta i giri interessati, per tutti i piloti
    gialle = rc[(rc["category"] == "Flag") & rc["flag"].isin(["YELLOW", "DOUBLE YELLOW"])]
    laps_stints = laps_stints[~laps_stints["lap_number"].isin(gialle["lap_number"].dropna().unique())].copy()
    registra("senza bandiere gialle", laps_stints)

    # Fase di ogni giro (asciutto / transizione / bagnato): toglie anche safety car e VSC, che rallentano tutti
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

    # Outlier: giri più lenti della mediana mobile (pilota, stint, fase) di oltre PCT_OUTLIER.
    def mediana_mobile(serie):
        return serie.rolling(MEDIAN_WINDOW, center=True, min_periods=MIN_PERIODS).median()

    laps_stints = laps_stints.sort_values(["driver_number", "stint_number", "lap_number"])
    laps_stints["rolling_median"] = laps_stints.groupby(["driver_number", "stint_number", "fase_id"])[
        "lap_duration"].transform(mediana_mobile)
    laps_stints["scostamento"] = laps_stints["lap_duration"] - laps_stints["rolling_median"]
    laps_stints["outlier"] = laps_stints["scostamento"] > laps_stints["rolling_median"] * PCT_OUTLIER
    giri_puliti = laps_stints[~laps_stints["outlier"]].copy()
    registra("non outlier", giri_puliti)
    return giri_puliti, pd.DataFrame(funnel, columns=["regola", "righe"])


# =============================================================================
# Perdita per sosta (sezione 12)
# =============================================================================

def numero_giri(laps):
    """Giri della gara: l'ultimo giro con almeno un tempo. OpenF1 aggiunge a volte una riga
    dopo il traguardo senza tempo (Monza 2023 ha 51 giri, ma lap_number arriva a 52)."""
    return int(laps.loc[laps["lap_duration"].notna(), "lap_number"].max())


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
# Modello del degrado per il simulatore (sezioni 8-9)
# =============================================================================

class ModelloDegrado:
    """tempo_giro = passo[mescola] + degrado[mescola] * età + carburante * giro

    Lineare di proposito: ogni coefficiente ha un significato (s, s/giro di età, s/giro di gara),
    il tempo cresce sempre con l'età se il degrado è positivo, e i controlli di buon senso
    si fanno cambiando un numero.
    """

    def __init__(self, passo, degrado, carburante):
        self.passo, self.degrado, self.carburante = dict(passo), dict(degrado), carburante

    @property
    def mescole(self):
        return list(self.passo)

    @classmethod
    def allena(cls, giri, mescole):
        giri = giri[giri["compound"].isin(mescole)]
        X = pd.DataFrame(index=giri.index)
        for m in mescole:
            e = (giri["compound"] == m).astype(float)
            X[f"passo_{m}"] = e
            X[f"degrado_{m}"] = e * giri["tyre_age"]
        X["carburante"] = giri["lap_number"]
        reg = LinearRegression(fit_intercept=False).fit(X, giri["lap_duration"])
        coef = dict(zip(X.columns, reg.coef_))
        return cls({m: coef[f"passo_{m}"] for m in mescole},
                   {m: coef[f"degrado_{m}"] for m in mescole},
                   coef["carburante"])

    def predict(self, mescola, eta, giro):
        return self.passo[mescola] + self.degrado[mescola] * np.asarray(eta) + self.carburante * np.asarray(giro)

    def con_degrado(self, nuovi):
        """Copia del modello con alcuni degradi cambiati (per i controlli)."""
        return ModelloDegrado(self.passo, {**self.degrado, **nuovi}, self.carburante)


def mescole_stimabili(giri):
    conteggi = giri[giri["compound"].isin(MESCOLE_ASCIUTTE)].groupby("compound").agg(
        giri=("lap_duration", "size"), piloti=("driver_number", "nunique"))
    ok = conteggi[(conteggi["giri"] >= MIN_GIRI_MESCOLA) & (conteggi["piloti"] >= MIN_PILOTI_MESCOLA)]
    return [m for m in MESCOLE_ASCIUTTE if m in ok.index], conteggi


# =============================================================================
# Simulatore (sezione 12)
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
# Controlli di buon senso (sezione 12)
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
# Back-testing (sezione 12)
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


def backtest(laps, stints, rc, giri_puliti, modello, n_giri, perdita_sosta, ottimo):
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
        return bt, np.nan
    bt["gap_ottimo"] = bt["tempo_sim"] - ottimo["tempo_totale"]
    bt["errore_sim"] = bt["sim_dal_giro_2"] - bt["reale_dal_giro_2"]

    # Errore del modello sui soli giri puliti (senza traffico, SC, soste)
    g = giri_puliti[giri_puliti["compound"].isin(modello.mescole)]
    previsto = np.array([modello.predict(m, a, l) for m, a, l in zip(g["compound"], g["tyre_age"], g["lap_number"])])
    mae_giri_puliti = float(np.mean(np.abs(previsto - g["lap_duration"])))
    return bt, mae_giri_puliti


# =============================================================================
# Validazione "una gara fuori" (sezione 11)
# =============================================================================

def validazione_una_gara_fuori(tutti):
    dati = tutti[tutti["compound"].isin(MESCOLE_ASCIUTTE)].copy()
    # Tempi relativi: secondi rispetto alla mediana dei giri puliti della stessa gara
    dati["tempo_rel"] = dati["lap_duration"] - dati.groupby("session_key")["lap_duration"].transform("median")
    for m in MESCOLE_ASCIUTTE:
        dati[f"eta_{m}"] = dati["tyre_age"] * (dati["compound"] == m)
    numeriche = ["tyre_age", "lap_number"] + [f"eta_{m}" for m in MESCOLE_ASCIUTTE]
    X, y, gruppi = dati[["compound"] + numeriche], dati["tempo_rel"], dati["session_key"]

    def pipeline(modello):
        pre = ColumnTransformer([("comp", OneHotEncoder(handle_unknown="ignore"), ["compound"])],
                                remainder="passthrough")
        return Pipeline([("pre", pre), ("m", modello)])

    modelli = {
        "baseline (mediana)": DummyRegressor(strategy="median"),
        "lineare (passo + degrado per mescola)": pipeline(LinearRegression()),
        "gradient boosting": pipeline(HistGradientBoostingRegressor(random_state=0)),
    }
    cv = LeaveOneGroupOut()
    righe = []
    for nome, m in modelli.items():
        mae = -cross_val_score(m, X, y, groups=gruppi, cv=cv, scoring="neg_mean_absolute_error")
        righe.append({"modello": nome, **{f"fuori {g}": round(v, 3) for g, v in zip(sorted(gruppi.unique()), mae)},
                      "MAE medio": round(mae.mean(), 3)})
    return pd.DataFrame(righe)


# =============================================================================
# Grafico del degrado per gara
# =============================================================================

def grafico_degrado(tabella, percorso):
    gare = list(dict.fromkeys(tabella["gara"]))
    larghezza = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, m in enumerate(MESCOLE_ASCIUTTE):
        t = tabella[tabella["mescola"] == m].set_index("gara").reindex(gare)
        x = np.arange(len(gare)) + (i - 1) * larghezza
        barre = ax.bar(x, t["degrado_s_giro"], width=larghezza - 0.02, color=COLORI_MESCOLA[m], label=m,
                       edgecolor="white", linewidth=1)
        ax.bar_label(barre, labels=[f"{v:.3f}" if pd.notna(v) else "" for v in t["degrado_s_giro"]],
                     fontsize=8, color="#444444", padding=2)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(np.arange(len(gare)), gare)
    ax.set_ylabel("degrado stimato (s per giro di età)")
    ax.set_title("Degrado per gara e mescola (modello lineare, giri puliti)", loc="left")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    ax.set_ylim(top=tabella["degrado_s_giro"].max() * 1.2)
    ax.legend(frameon=False, ncols=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(percorso, dpi=150)
    plt.close(fig)


# =============================================================================
# Programma principale
# =============================================================================

def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    sessioni = pd.read_csv(os.path.join(DATA_DIR, "sessions.csv")).set_index("session_key")
    circuito = sessioni["circuit_short_name"].to_dict()

    # --- 1. Gare asciutte e pulizia --------------------------------------------
    gare, tabelle = {}, []
    for session_key in GARE_CANDIDATE:
        dati = carica_gara(session_key)
        if dati is None:
            continue
        puliti, funnel = prepara_gara(dati["laps"], dati["stints"], dati["weather"], dati["rc"])
        puliti["session_key"] = session_key
        print(f"\n--- {circuito.get(session_key, session_key)} ({session_key}): tabella degli scarti")
        print(funnel.to_string(index=False))
        gare[session_key] = {**dati, "puliti": puliti}
        tabelle.append(puliti)

    if not tabelle:
        print("Nessuna gara disponibile.")
        return
    tutti = pd.concat(tabelle, ignore_index=True)
    assert len(tutti) == sum(len(t) for t in tabelle)
    assert not tutti[["compound", "tyre_age", "lap_number", "lap_duration"]].isna().any().any(), "NaN nelle feature"

    # --- 2-4. Modello, perdita, strategie, controlli, back-testing per ogni gara --
    righe_degrado, riepilogo = [], []
    for session_key, g in gare.items():
        nome = circuito.get(session_key, session_key)
        laps = segna_soste(g["laps"])
        puliti = g["puliti"]
        n_giri = numero_giri(laps)

        mescole, conteggi = mescole_stimabili(puliti)
        escluse = [m for m in conteggi.index if m not in mescole]
        modello = ModelloDegrado.allena(puliti, mescole)
        eta_max = puliti[puliti["compound"].isin(mescole)].groupby("compound")["tyre_age"].max().to_dict()
        pit = stima_perdita_pit(laps, puliti, giri_neutralizzati(laps, g["rc"]))

        print(f"\n================ {nome} ({session_key}), {n_giri} giri ================")
        print(f"Perdita per sosta: {pit['perdita']:.1f} s  (soste usate {pit['soste_usate']}, "
              f"scartate perché sporche/incomplete {pit['soste_scartate']})")
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

        bt, mae_giri = backtest(laps, g["stints"], g["rc"], puliti, modello, n_giri, pit["perdita"], ottimo)
        if bt.empty:
            print("Back-testing: nessun pilota con strategia simulabile.")
            continue
        print("Back-testing (primi 10 arrivati con strategia simulabile):")
        print(bt.head(10)[["pos", "pilota", "piano", "tempo_sim", "gap_ottimo", "reale_dal_giro_2", "errore_sim"]]
              .round(1).to_string(index=False))
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

    # --- Degrado per gara: tabella e grafico (sezione 11) ---------------------
    tabella = pd.DataFrame(righe_degrado)
    tabella.to_csv(os.path.join(DATA_DIR, "degrado_per_gara.csv"), index=False)
    grafico_degrado(tabella, os.path.join(FIG_DIR, "degrado_per_gara.png"))
    print("\n================ Degrado per gara e mescola ================")
    print(tabella.round(3).to_string(index=False))

    print("\n================ Riepilogo strategie ================")
    print(pd.DataFrame(riepilogo).to_string(index=False))

    # --- Validazione "una gara fuori" (sezione 11) ----------------------------
    print("\n================ Validazione una gara fuori (MAE in s, tempi relativi alla mediana di gara) ================")
    print(validazione_una_gara_fuori(tutti).to_string(index=False))


if __name__ == "__main__":
    main()
