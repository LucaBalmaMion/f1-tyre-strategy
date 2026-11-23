import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = 'https://api.openf1.org/v1'
DATA_DIR = 'data'
SESSION_KEY = 9094

os.makedirs(DATA_DIR, exist_ok=True)

def scarica(endpoint, params, nome_file):
    """Scarica una tabella da OpenF1, o la legge da file se l'ho già scaricata."""
    percorso = os.path.join(DATA_DIR, nome_file)
    if os.path.exists(percorso):
        return pd.read_csv(percorso)
    risposta = requests.get(BASE_URL + '/' + endpoint, params=params)
    risposta.raise_for_status()
    dati = risposta.json()
    dt = pd.DataFrame(dati)
    column_list = [c for c in dt.columns if dt[c].apply(lambda x: isinstance(x, list)).any()]
    dt = dt.drop(columns=column_list, errors='ignore')
    dt.to_csv(percorso, index=False)
    time.sleep(1)
    return dt



dt_sessions = scarica('sessions', {'year': 2023, 'session_name': 'Race'}, 'sessions.csv')
dt_laps = scarica('laps', {'session_key' : 9094}, 'laps.csv')
dt_stints = scarica('stints', {'session_key' : 9094}, 'stints.csv')
dt_race_control = scarica('race_control', {'session_key' : 9094}, 'race_control.csv')
dt_weather = scarica('weather', {'session_key' : 9094}, 'weather.csv')

PILOTA = 1

giri_pilota = dt_laps[(dt_laps['driver_number'] == PILOTA) & (dt_laps['lap_number'].between(2,51)) ]   # pilota 1, giri tra 2 e 51
giri_pilota = giri_pilota[giri_pilota['is_pit_out_lap'] == False]                  # togli i giri di uscita dai box
giri_pilota = giri_pilota.dropna(subset=['lap_duration'])    # togli i tempi mancanti

print(giri_pilota.shape)


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(giri_pilota['lap_number'], giri_pilota['lap_duration'], marker='o')
ax.set_xlabel('Numero di giri')
ax.set_ylabel('Durata del giro (s)')
ax.set_title('Grafico dei giri 2 - 51')
ax.grid(True)
fig.savefig('figures/01_tempo_giro_pilota1.png', dpi=150)
plt.show()








