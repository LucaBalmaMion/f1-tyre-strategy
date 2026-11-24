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
    percorso = os.path.join(DATA_DIR, nome_file)
    if os.path.exists(percorso):
        return pd.read_csv(percorso)
    risposta = requests.get(BASE_URL + '/' + endpoint, params=params)
    risposta.raise_for_status()
    dati = risposta.json()
    df = pd.DataFrame(dati)
    column_list = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, list)).any()]
    df = df.drop(columns=column_list, errors='ignore')
    df.to_csv(percorso, index=False)
    time.sleep(1)
    return df


df_sessions = scarica('sessions', {'year': 2023, 'session_name': 'Race'}, 'sessions.csv')
df_laps = scarica('laps', {'session_key' : 9094}, 'laps.csv')
df_stints = scarica('stints', {'session_key' : 9094}, 'stints.csv')
df_race_control = scarica('race_control', {'session_key' : 9094}, 'race_control.csv')
df_weather = scarica('weather', {'session_key' : 9094}, 'weather.csv')

PILOTA = 1

giri_pilota = df_laps[(df_laps['driver_number'] == PILOTA) & (df_laps['lap_number'].between(2,51)) ]
giri_pilota = giri_pilota[giri_pilota['is_pit_out_lap'] == False]
giri_pilota = giri_pilota.dropna(subset=['lap_duration'])


'''
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(giri_pilota['lap_number'], giri_pilota['lap_duration'], marker='o')
ax.set_xlabel('Numero di giri')
ax.set_ylabel('Durata del giro (s)')
ax.set_title('Grafico dei giri 2 - 51')
ax.grid(True)
fig.savefig('figures/01_tempo_giro_pilota1.png', dpi=150)
plt.show()

print(df_laps.columns)
print(df_laps['lap_duration'].describe().round(3))
df_laps['lap_duration'] = df_laps['lap_duration'] < 90
print(df_laps['lap_duration'].head(15))

print(df_laps.loc[df_laps['driver_number'] == PILOTA, ['lap_number', 'lap_duration']])
df_laps.groupby('driver_number')['lap_duration'].mean()
df_laps.groupby('driver_number')['lap_duration'].agg(['count', 'mean', 'min', 'max'])
'''

#print(df_laps.columns)
df_laps = df_laps.sort_values(['driver_number', 'lap_number'])

#print(df_laps['driver_number'].value_counts())
#print(df_laps.sort_values(by='lap_duration', ascending=True)[['driver_number', 'lap_duration']].head(5))
#print(df_laps[(df_laps['lap_number'] != 1) & (df_laps['is_pit_out_lap'] == False)].groupby('driver_number')['lap_duration'].median())
#print(df_laps['lap_duration'].isna().sum())
#print(df_laps[df_laps['lap_duration'].isna()]['lap_number'])
#df_laps['is_pit_in_lap'] = df_laps.groupby(['driver_number'])['is_pit_out_lap'].shift(-1)
#print(df_laps[df_laps['lap_number'].between(53,60)][['driver_number', 'lap_number', 'is_pit_in_lap', 'is_pit_out_lap']])

df_laps_stints = df_laps.merge(df_stints, on=['driver_number', 'meeting_key', 'session_key',])
df_laps_stints = df_laps_stints[(df_laps_stints['lap_number'] >= df_laps_stints['lap_start']) & (df_laps_stints['lap_number'] <= df_laps_stints['lap_end'])]
df_laps_stints['tyre_age'] = df_laps_stints['tyre_age_at_start'] + df_laps_stints['lap_number'] - df_laps_stints['lap_start']

df_laps_stints['date_start'] = pd.to_datetime(df_laps_stints['date_start'], format='ISO8601')
df_weather['date'] = pd.to_datetime(df_weather['date'], format='ISO8601')
df_laps_stints = df_laps_stints.sort_values('date_start')
df_weather = df_weather.sort_values('date')
df_laps_stints = pd.merge_asof(df_laps_stints, df_weather, left_on='date_start', right_on='date')

df_yellow = df_race_control[(df_race_control['category'] == 'Flag') & (df_race_control['flag'].isin(['YELLOW', 'DOUBLE YELLOW']))]

laps_flag = df_yellow['lap_number'].unique()
df_laps_stints = df_laps_stints[~df_laps_stints['lap_number'].isin(laps_flag)]

df_laps_stints = df_laps_stints.sort_values(['driver_number','stint_number', 'lap_number'])

df_laps_stints['rolling_median'] = (
    df_laps_stints
    .groupby(['driver_number', 'stint_number'])['lap_duration']
    .transform(
        lambda x: x.rolling(
            window=7,
            center=True,
            min_periods=5
        ).median()
    )
)

df_laps_stints['scostamento'] = df_laps_stints['lap_duration'] - df_laps_stints['rolling_median']

df_laps_stints['outlier'] = (df_laps_stints['scostamento'] > 2.5)

#print(df_laps_stints[df_laps_stints['outlier'] == True][['driver_number', 'lap_number', 'lap_duration', 'is_pit_out_lap', 'stint_number', 'date_start']].sort_values(['lap_number', 'driver_number']))
print(
    df_laps_stints[df_laps_stints['outlier']]
    .groupby('lap_number')
    .agg(
        numero_outlier=('driver_number', 'count'),
        durata_media=('lap_duration', 'mean'),
        durata_max=('lap_duration', 'max'),
        durata_min=('lap_duration', 'min')
    )
)


