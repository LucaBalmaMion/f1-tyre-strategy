import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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
SESSION_KEY, YEAR, SESSION_NAME = 9094, 2023, "Race"
# Soglie della pulizia
GIRO_FINE_ASCIUTTO_RIF = 49   # ultimo giro per il riferimento in asciutto (Monaco 2023)
SOGLIA_TRANSIZIONE, SOGLIA_BAGNATO = 2, 6     # secondi sopra il tempo asciutto
FINESTRA_MEDIANA, MIN_PERIODI, SOGLIA_OUTLIER = 7, 5, 2
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


df_sessions = scarica("sessions", {"year": YEAR, "session_name": SESSION_NAME}, "sessions.csv")
df_laps = scarica("laps", {"session_key": SESSION_KEY}, "laps.csv")
df_stints = scarica("stints", {"session_key": SESSION_KEY}, "stints.csv")
df_race_control = scarica("race_control", {"session_key": SESSION_KEY}, "race_control.csv")
df_weather = scarica("weather", {"session_key": SESSION_KEY}, "weather.csv")

# Giri: giro di rientro = giro prima di un'uscita (calcolato prima di togliere righe)
df_laps = df_laps.copy()
df_laps["date_start"] = pd.to_datetime(df_laps["date_start"], format="ISO8601", utc=True).dt.as_unit("ns")
df_laps["is_pit_out_lap"] = df_laps["is_pit_out_lap"].eq(True)
df_laps = df_laps.sort_values(["driver_number", "lap_number"])
df_laps["is_pit_in_lap"] = df_laps.groupby("driver_number")["is_pit_out_lap"].shift(-1, fill_value=False)
registra("giri scaricati", df_laps)

df_laps = df_laps.dropna(subset=["lap_duration", "date_start"])
registra("tempo e data presenti", df_laps)

df_laps = df_laps[~df_laps["is_pit_out_lap"]]
registra("non giro di uscita", df_laps)

df_laps = df_laps[~df_laps["is_pit_in_lap"]]
registra("non giro di rientro", df_laps)

df_laps = df_laps[df_laps["lap_number"] > 1]
registra("non giro 1", df_laps)

# Giri + stint: ogni giro prende lo stint che lo contiene, poi si calcola l'età della gomma
df_laps_stints = df_laps.merge(df_stints, on=["driver_number", "meeting_key", "session_key"])
df_laps_stints = df_laps_stints[df_laps_stints["lap_number"].between(df_laps_stints["lap_start"], df_laps_stints["lap_end"])].copy()
registra("giri con uno stint", df_laps_stints)

df_laps_stints["tyre_age"] = df_laps_stints["tyre_age_at_start"] + df_laps_stints["lap_number"] - df_laps_stints["lap_start"]

# Meteo: ultima misura precedente all'inizio del giro (senza le chiavi già presenti nei giri)
df_weather = df_weather.drop(columns=["meeting_key", "session_key"])
df_weather["date"] = pd.to_datetime(df_weather["date"], format="ISO8601", utc=True).dt.as_unit("ns")
df_laps_stints = pd.merge_asof(df_laps_stints.sort_values("date_start"), df_weather.sort_values("date"),
                               left_on="date_start", right_on="date", direction="backward")

# Bandiere gialle: scarta i giri interessati, per tutti i piloti
gialle = df_race_control[(df_race_control["category"] == "Flag") & df_race_control["flag"].isin(["YELLOW", "DOUBLE YELLOW"])]
df_laps_stints = df_laps_stints[~df_laps_stints["lap_number"].isin(gialle["lap_number"].dropna().unique())].copy()
registra("senza bandiere gialle", df_laps_stints)

# Fase meteo di ogni giro (asciutto / transizione / bagnato) e sua numerazione
riepilogo_giri = df_laps_stints.groupby("lap_number").agg(tempo_mediano=("lap_duration", "median"), quota_pioggia=("rainfall", "mean")).reset_index()
asciutti = riepilogo_giri[(riepilogo_giri["lap_number"] <= GIRO_FINE_ASCIUTTO_RIF) & (riepilogo_giri["quota_pioggia"] == 0)]
tempo_asciutto = asciutti["tempo_mediano"].median()


def classifica_fase(riga):
    if riga["tempo_mediano"] >= tempo_asciutto + SOGLIA_BAGNATO:
        return "bagnato"
    if riga["quota_pioggia"] > 0 or riga["tempo_mediano"] >= tempo_asciutto + SOGLIA_TRANSIZIONE:
        return "transizione"
    return "asciutto"


riepilogo_giri["fase_meteo"] = riepilogo_giri.apply(classifica_fase, axis=1)
riepilogo_giri["fase_id"] = riepilogo_giri["fase_meteo"].ne(riepilogo_giri["fase_meteo"].shift()).cumsum()
df_laps_stints = df_laps_stints.merge(riepilogo_giri[["lap_number", "fase_meteo", "fase_id"]], on="lap_number", how="left")
df_laps_stints = df_laps_stints[df_laps_stints["fase_meteo"] == "asciutto"].copy()
registra("solo fase asciutta", df_laps_stints)


# Outlier: giri più lenti di SOGLIA_OUTLIER secondi rispetto alla mediana mobile di pilota, stint e fase.
# Dove la mediana non è calcolabile (primo/ultimo giro del gruppo) lo scostamento è NaN: il giro non viene segnalato.
def mediana_mobile(serie):
    return serie.rolling(FINESTRA_MEDIANA, center=True, min_periods=MIN_PERIODI).median()


df_laps_stints = df_laps_stints.sort_values(["driver_number", "stint_number", "lap_number"])
df_laps_stints["rolling_median"] = df_laps_stints.groupby(["driver_number", "stint_number", "fase_id"])["lap_duration"].transform(mediana_mobile)
df_laps_stints["scostamento"] = df_laps_stints["lap_duration"] - df_laps_stints["rolling_median"]
df_laps_stints["outlier"] = df_laps_stints["scostamento"] > SOGLIA_OUTLIER
giri_puliti = df_laps_stints[~df_laps_stints["outlier"]].copy()
registra("non outlier", giri_puliti)
giri_puliti.to_csv(os.path.join(DATA_DIR, "giri_puliti.csv"), index=False)

#GRAFICI

SAVE_PLOT = False
out_dir = Path(__file__).resolve().parent / "figures"
if SAVE_PLOT:

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_hist, ax_hist = plt.subplots()

    #istogramma
    ax_hist.hist(df_laps_stints['lap_duration'], bins=80)
    ax_hist.set_xlabel('Tempo sul giro (s)')
    ax_hist.set_ylabel('Numero di giri')
    fig_hist.savefig(out_dir / "istogramma_lap_duration.png", dpi=300, bbox_inches="tight")


    #boxplot
    fig_box, ax_box = plt.subplots()
    df_laps_stints.boxplot(column='lap_duration', by='compound', ax=ax_box)
    fig_box.savefig(out_dir / "boxplot_lap_duration_compound.png", dpi=300, bbox_inches="tight")



    #scatter
    fig_scatter, ax_scatter = plt.subplots()
    for comp, g in df_laps_stints.groupby('compound'):
        ax_scatter.scatter(g['tyre_age'], g['lap_duration'], label=comp, s=12)
    ax_scatter.set_xlabel('Età gomma (giri)')
    ax_scatter.set_ylabel('Tempo (s)')
    ax_scatter.legend()
    fig_scatter.savefig(out_dir / "scatter_tyre_age_lap_duration.png", dpi=300, bbox_inches="tight")



    colori_mescola = {
        'SOFT': 'tab:red',
        'MEDIUM': 'goldenrod',
        'HARD': 'tab:gray',
    }

    piloti = sorted(df_laps_stints['driver_number'].unique())
    fig1, axs = plt.subplots(4, 5, figsize=(14, 9), sharex=True, sharey=True)

    for ax, d in zip(axs.flat, piloti):
        g = df_laps_stints[df_laps_stints['driver_number'] == d].sort_values('lap_number')

        for i in range(len(g) - 1):
            x_seg = g['lap_number'].iloc[i:i + 2]
            y_seg = g['lap_duration'].iloc[i:i + 2]
            comp = g['compound'].iloc[i]
            ax.plot(
                x_seg,
                y_seg,
                color=colori_mescola.get(comp, 'tab:gray'),
                linewidth=1.5,
            )

        ax.set_title(f'Pilota {d}')
        ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color=col, lw=2, label=comp)
        for comp, col in colori_mescola.items()
    ]
    fig1.legend(handles=handles, title='Compound', loc='upper right')
    fig1.supxlabel('Numero di giro')
    fig1.supylabel('Tempo sul giro (s)')
    fig1.tight_layout()
    fig1.savefig(out_dir / "piloti_lap_duration_compound.png", dpi=300, bbox_inches="tight")

    # istogramma 2
    fig_hist2, ax_hist2 = plt.subplots()

    ax_hist2.hist(df_laps_stints['scostamento'], bins=30)
    ax_hist2.set_xlabel('Tempo sul giro (s)')
    ax_hist2.set_ylabel('Numero di giri')
    fig_hist2.savefig(out_dir / "istogramma_scostamento.png", dpi=300, bbox_inches="tight")

'''
##**PRIMO MODELLO**##
#regressione lineare
SELECTED_COMPOUND = ['HARD']

dati = giri_puliti[giri_puliti['compound'].isin(SELECTED_COMPOUND)]
X = dati[['tyre_age', 'lap_number']]
y = dati['lap_duration']
gruppi = dati['driver_number']

modello = LinearRegression().fit(X, y)
print(modello.coef_, modello.intercept_)

cv = GroupKFold(n_splits=dati['driver_number'].nunique())
for nome, mod in [('baseline', DummyRegressor(strategy='mean')), ('lineare', LinearRegression())]:
    punteggi = cross_val_score(mod, X, y, groups=gruppi, cv=cv, scoring='neg_mean_absolute_error')
    mae_medio = -punteggi.mean()
    mae_std = punteggi.std()
    #print(f"{nome}: {mae_medio:.3f} s (+/- {mae_std:.3f} s)")
    '''


## CONFRONTO TRA MODELLI ##
dati = giri_puliti.copy()


gruppi = dati['driver_number']
piloti_train = gruppi.unique()[:-5]
tr = gruppi.isin(piloti_train)
X = dati[['tyre_age', 'lap_number', 'compound', 'track_temperature', 'humidity', 'wind_speed']]
y = dati['lap_duration']



'''
#pipeline----
pre = ColumnTransformer([('comp', OneHotEncoder(handle_unknown='ignore'), ['compound'])], remainder='passthrough')
modelli = {
    'baseline (media)': DummyRegressor(strategy='mean'),
    'lineare': Pipeline([('pre', pre), ('m', LinearRegression())]),
    'random forest 5': Pipeline([('pre', pre), ('m', RandomForestRegressor(n_estimators=200, random_state=0, min_samples_leaf=5))]),
    'hist gradient boosting d6': Pipeline([('pre', pre), ('m', HistGradientBoostingRegressor(random_state=0, max_depth=6, min_samples_leaf=5))]),
}
cv = GroupKFold(n_splits=dati['driver_number'].nunique())
#righe = []
for nome, mod in modelli.items():
    s = cross_val_score(mod, X, y, groups=gruppi, cv=cv, scoring='neg_mean_absolute_error')
    #righe.append({'nome' : nome, 'media' : -s.mean(), 'deviazione' : s.std()})
    #print(f'{nome:24s} MAE medio {-s.mean():.3f} MAE std {s.std():.3f} (fold: {np.round(-s, 3)})')

pi = permutation_importance(modelli['hist gradient boosting d6'], X, y, n_repeats=5, random_state=0)
for col, val in zip(X.columns, np.round(pi.importances_mean, 3)):
    print(f'{col:20s} {val:+.3f}')
'''
'''
#SEZIONE 10
pre = ColumnTransformer([('comp', OneHotEncoder(handle_unknown='ignore'), ['compound'])], remainder='passthrough')
mod_pi = Pipeline([('pre', pre), ('m', HistGradientBoostingRegressor(random_state=0, max_depth=6, min_samples_leaf=5))])
mod_pi.fit(X[tr], y[tr])

pi = permutation_importance(mod_pi, X[~tr], y[~tr], n_repeats=5, random_state=0, scoring='neg_mean_absolute_error')
for col, m, s in zip(X.columns, pi.importances_mean, pi.importances_std):
    print(f"{col:20s} {m:+.3f} ± {s:.3f}")
'''


SAVE_PLOT = False
if SAVE_PLOT:
    pre = ColumnTransformer([('comp', OneHotEncoder(handle_unknown='ignore'), ['compound'])], remainder='passthrough')
    mod = Pipeline([('pre', pre), ('m', HistGradientBoostingRegressor(random_state=0, max_depth=6, min_samples_leaf=5))])
    cv = GroupKFold(n_splits=dati['driver_number'].nunique())
    y_pred = cross_val_predict(mod, X, y, cv=cv, groups=gruppi)
    scarto = y - y_pred
    fig_scatter2, ax_scatter2 = plt.subplots()
    for comp, g in dati.groupby('compound'):
        ax_scatter2.scatter(g['tyre_age'],scarto.loc[g.index], label=comp, s=12)
    ax_scatter2.set_xlabel('Età gomma (giri)')
    ax_scatter2.set_ylabel('Residuo (s)')
    ax_scatter2.legend()
    fig_scatter2.savefig(out_dir / "scatter_tyre_age_residuo.png", dpi=300, bbox_inches="tight")

print(dati[dati['lap_number'] == 16][['driver_number', 'outlier', 'lap_duration']].sort_values(by='lap_duration', ascending=False))