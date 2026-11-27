import pandas as pd

# Soglie della pulizia
PCT_TRANSITION, PCT_WET = 0.03, 0.08
MEDIAN_WINDOW, MIN_PERIODS, PCT_OUTLIER = 7, 5, 0.025


def segna_soste(laps):
    """Converte le date e aggiunge is_pit_in_lap (= giro prima di un giro di uscita)."""
    laps = laps.copy()
    laps["date_start"] = pd.to_datetime(laps["date_start"], format="ISO8601", utc=True).dt.as_unit("ns")
    laps["is_pit_out_lap"] = laps["is_pit_out_lap"].eq(True)
    laps = laps.sort_values(["driver_number", "lap_number"])
    laps["is_pit_in_lap"] = laps.groupby("driver_number")["is_pit_out_lap"].shift(-1, fill_value=False)
    return laps


def numero_giri(laps):
    """Giri della gara: l'ultimo giro con almeno un tempo. OpenF1 aggiunge a volte una riga
    dopo il traguardo senza tempo (Monza 2023 ha 51 giri, ma lap_number arriva a 52)."""
    return int(laps.loc[laps["lap_duration"].notna(), "lap_number"].max())


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
