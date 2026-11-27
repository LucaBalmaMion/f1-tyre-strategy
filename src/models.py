"""Modello del degrado, validazione "una gara fuori" e confronto con la baseline (sezioni 8-9 e 11)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MESCOLE_ASCIUTTE = ["SOFT", "MEDIUM", "HARD"]
MIN_GIRI_MESCOLA, MIN_PILOTI_MESCOLA = 50, 3   # sotto queste soglie la mescola non è stimabile
COLORI_MESCOLA = {"SOFT": "#2a78d6", "MEDIUM": "#eb6834", "HARD": "#1baf7a"}


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


def allena_gara(puliti):
    """Modello di una gara: (modello, mescole usate, conteggi per mescola, età massima per mescola)."""
    mescole, conteggi = mescole_stimabili(puliti)
    modello = ModelloDegrado.allena(puliti, mescole)
    eta_max = puliti[puliti["compound"].isin(mescole)].groupby("compound")["tyre_age"].max().to_dict()
    return modello, mescole, conteggi, eta_max


def mae_giri_puliti(modello, giri_puliti):
    """Errore medio del modello sui soli giri puliti (senza traffico, SC, soste)."""
    g = giri_puliti[giri_puliti["compound"].isin(modello.mescole)]
    previsto = np.array([modello.predict(m, a, l) for m, a, l in zip(g["compound"], g["tyre_age"], g["lap_number"])])
    return float(np.mean(np.abs(previsto - g["lap_duration"])))


def validazione_una_gara_fuori(tutti):
    """MAE di baseline, lineare e gradient boosting allenando su tutte le gare tranne una."""
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
