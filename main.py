import requests
import pandas as pd

def da_url_a_dataframe(url, params):
    risposta = requests.get(url, params=params)
    dati = risposta.json()
    dt = pd.DataFrame(dati)
    return dt

url_sessions = 'https://api.openf1.org/v1/sessions'
params_sessions = {'year': 2023, 'session_name': 'Race'}

url_laps = 'https://api.openf1.org/v1/laps'
params_laps = {'session_key' : 9094}

url_stints = 'https://api.openf1.org/v1/stints'
params_stints = {'session_key' : 9094}

url_race_control = 'https://api.openf1.org/v1/race_control'
params_race_control = {'session_key' : 9094}

url_weather = 'https://api.openf1.org/v1/weather'
params_weather = {'session_key' : 9094}


dt_sessions = da_url_a_dataframe(url_sessions, params_sessions)
dt_laps = da_url_a_dataframe(url_laps, params_laps)
dt_stints = da_url_a_dataframe(url_stints, params_stints)
dt_race_control = da_url_a_dataframe(url_race_control, params_race_control)
dt_weather = da_url_a_dataframe(url_weather, params_weather)


#print(dt_laps[dt_laps['driver_number'] == 1][['lap_number', 'lap_duration', 'is_pit_out_lap']].head(60))
#print(dt_race_control.columns)
print(dt_race_control[(dt_race_control['lap_number'] == 54)][['flag', 'message']])









