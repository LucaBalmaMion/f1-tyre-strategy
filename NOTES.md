# Diario del progetto

> Le sezioni 2-10 sono da completare con il modello dell'Appendice A della guida.

## Sezione 11: più gare

### Scelta delle gare
- Candidate: Sakhir (7953), Monte Carlo (9094), Catalunya (9102), Monza (9157), Mexico City (9181).
- **Monte Carlo esclusa automaticamente**: pioggia nel 15% delle misure meteo. Il filtro è `MAX_QUOTA_PIOGGIA = 0`
  (gara asciutta = nessuna misura con `rainfall`).
- **Catalunya**: download non riuscito il 27/11/2026 (l'API restituiva 401 per una sessione live in corso).
  Il programma la salta con un messaggio e la include da sola quando l'API torna disponibile.
- Mexico City ha una **bandiera rossa al giro 34**: la tengo, ma ne tengo conto nella perdita per sosta e nel back-testing.

### Tabella degli scarti
Il programma la stampa per ogni gara. Giri rimasti: Sakhir 843/1058, Monza 876/968, Mexico City 1080/1286.

### Degrado per gara e mescola (modello lineare, s per giro di età)
| Gara | Mescola | Degrado | Giri | Piloti |
|---|---|---|---|---|
| Sakhir | SOFT | 0.114 | 325 | 20 |
| Sakhir | HARD | 0.096 | 510 | 19 |
| Monza | MEDIUM | 0.065 | 402 | 19 |
| Monza | HARD | 0.060 | 474 | 19 |
| Mexico City | MEDIUM | 0.047 | 596 | 19 |
| Mexico City | HARD | 0.054 | 477 | 19 |

Grafico: `figures/degrado_per_gara.png`. Dati: `data/degrado_per_gara.csv`.

- Sakhir ha il degrado più alto (asfalto molto abrasivo, caldo): è coerente con quello che si sa del circuito.
- Le mescole con meno di 50 giri puliti o meno di 3 piloti sono escluse perché la stima non è affidabile
  (MEDIUM a Sakhir: 8 giri; SOFT a Mexico City: 7 giri).
- Attenzione: "SOFT/MEDIUM/HARD" non sono la stessa gomma su circuiti diversi (sezione 5.4),
  quindi il confronto tra gare va fatto per circuito, non per nome della mescola.

### Validazione "una gara fuori" (MAE in s, tempi relativi alla mediana della gara)
| Modello | fuori Sakhir | fuori Monza | fuori Mexico | medio |
|---|---|---|---|---|
| baseline (mediana) | 1.012 | 0.599 | 0.857 | 0.823 |
| lineare (passo + degrado per mescola) | 0.871 | 0.730 | 0.708 | 0.769 |
| gradient boosting | 0.827 | 0.773 | 0.800 | 0.800 |

- In media il lineare batte la baseline, ma **non su Monza**: il degrado imparato sulle altre gare (più alto) non si trasferisce.
  Il degrado dipende dal circuito, ed è per questo che il simulatore usa un modello **per gara**.
- Il gradient boosting non fa meglio del lineare: con questi dati la complessità in più non serve.

## Sezione 12: simulatore

### Modello usato dal simulatore
`tempo = passo[mescola] + degrado[mescola] × età + carburante × giro`, un modello per gara.
Prima usavo un HistGradientBoosting con meteo, ma la curva prevista non era monotona:
a Monte Carlo una gomma di 40 giri risultava più veloce di una di 15, e l'ottimizzatore sfruttava quel rumore.
Il modello lineare ha tre vantaggi: ogni coefficiente ha un significato, il tempo cresce sempre con l'età
e i controlli di buon senso si fanno cambiando un numero.

### Convenzione sull'età
Età 0 nel primo giro dello stint, sia nei dati (`tyre_age_at_start + lap_number - lap_start`) sia nel simulatore.
Uno stint di L giri arriva a età L-1: il limite è `L - 1 ≤ età massima osservata`.

### Numero di giri
`n_giri` = ultimo giro con almeno un tempo registrato. OpenF1 aggiunge una riga dopo il traguardo
(il rientro ai box), quindi `lap_number.max()` dava 52 giri per Monza (sono 51) e 72 per Mexico City (sono 71).

### Perdita per sosta
- Riferimento: mediana degli ultimi 5 giri puliti del pilota prima della sosta (almeno 3).
- `perdita = (rientro − rif) + (uscita − rif)`, poi la mediana su tutte le soste.
- **Soste scartate**: quelle con il giro di rientro o di uscita "neutralizzato", cioè con bandiere gialle o rosse,
  un messaggio di safety car, oppure un tempo mediano del gruppo più lento del 3% rispetto al passo gara
  (cattura SC, VSC e bandiera rossa). Senza questo filtro Mexico City dava 95.6 s (bandiera rossa)
  e Monte Carlo 62.3 s (pioggia).
- Risultati: Sakhir 23.3 s (37 soste usate, 13 scartate), Monza 24.1 s (25 / 0), Mexico City 21.4 s (17 / 21).

### Strategie ottime (regola delle due mescole, stint ≥ 5 giri, età ≤ massima osservata)
| Gara | 1 sosta | 2 soste | Ottima |
|---|---|---|---|
| Sakhir | impossibile (HARD max 28, SOFT max 23 giri, gara di 57) | SOFT(17) > HARD(37) > HARD | 2 soste |
| Monza | MEDIUM(24) > HARD, 4456.1 s | +9.7 s | 1 sosta |
| Mexico City | MEDIUM(38) > HARD, 5989.7 s | HARD(21) > MEDIUM(46) > MEDIUM, −1.1 s | praticamente pari |

A Mexico City 1 e 2 soste differiscono di 1.1 s: molto meno dell'errore del modello, quindi è un pareggio.

### Controlli di buon senso
Tutti e 4 superati in tutte e 3 le gare (vengono eseguiti a ogni lancio, senza limite di età):
1. la perdita per sosta (15/20/25 s) non sposta il giro di sosta e aggiunge al tempo esattamente la differenza;
2. "2 soste − 1 sosta" cresce di esattamente 10 s quando la perdita passa da 10 a 20 a 30 s;
3. con degrado zero, 2 soste = 1 sosta + perdita (20.00 s);
4. raddoppiando il degrado di una mescola usata, il tempo peggiora e il piano cambia.

### Back-testing
| Gara | Primo arrivato: gap dall'ottimo | Errore modello (giri puliti) | Errore sim. vs reale (gara intera, mediana) | Differenza tra strategie reali (10°-90° perc.) |
|---|---|---|---|---|
| Sakhir | +9.3 s | 0.62 s/giro | 24.4 s | 7.4 s |
| Monza | +1.0 s | 0.48 s/giro | 25.4 s | 15.5 s |
| Mexico City | −8.7 s | 0.69 s/giro | n.d. (bandiera rossa) | 25.4 s |

- A Monza il vincitore (MEDIUM fino al giro 20, poi HARD) è a 1 s dall'ottimo: strategia praticamente ottima.
- A Mexico City il vincitore risulta più veloce dell'"ottimo" perché ha cambiato gomme gratis durante la bandiera rossa
  (nel back-testing quelle soste costano 0). Gli stint spezzati da OpenF1 alla ripartenza vengono riuniti.
- **L'errore sulla gara intera (~25 s) è più grande della differenza tra strategie (7-15 s)**.
  Gran parte di quell'errore però è il passo del pilota: il modello descrive un pilota "medio",
  i primi sono più veloci e gli ultimi più lenti. Lo scarto di passo è quasi uguale con qualsiasi strategia,
  quindi il confronto **tra strategie per lo stesso pilota** è più affidabile del tempo assoluto. Resta però un limite.

### Cosa il simulatore non sa (da riportare nel README)
- **Traffico**: simula un pilota che gira da solo.
- **Safety car e bandiere**: una sosta in regime di safety car costa meno; il simulatore usa sempre la stessa perdita
  (tranne la bandiera rossa nel back-testing).
- **Meteo**: il modello è allenato solo su gare e fasi asciutte.
- **Carburante**: il numero di giro è un sostituto e contiene anche l'evoluzione della pista.
- **Gomme fredde**: il giro di uscita più lento è assorbito nella perdita per sosta.
- **Gomme usate**: il simulatore parte sempre da gomme nuove.
- **Undercut e overcut**: richiederebbero di simulare due piloti insieme.
- **Passo del pilota**: c'è un solo pilota medio per gara.

## Sezione 13: struttura del repository

### Da `main.py` a moduli e script
Tutto il codice stava in un unico `main.py`. Ora è diviso così:

| File | Cosa contiene |
|---|---|
| `src/data.py` | `scarica` (cache su CSV), `carica_gara` (salta le gare con pioggia o non scaricabili), percorsi e gare candidate |
| `src/clean.py` | `segna_soste`, `prepara_gara` (regole di pulizia + tabella degli scarti), `numero_giri`, soglie |
| `src/models.py` | `ModelloDegrado`, `mescole_stimabili`, `allena_gara`, `mae_giri_puliti`, validazione una gara fuori, grafico |
| `src/simulate.py` | perdita per sosta, `migliore_strategia`, controlli di buon senso, back-testing |
| `scripts/01..04` | i quattro passi, da lanciare in ordine |
| `notebooks/journey.py` | la prima esplorazione su Monte Carlo (ex `journey.py` nella radice) |

- Gli script si passano i risultati **tramite file** in `data/`: lo 02 scrive `giri_puliti_<gara>.csv` e
  `scarti_<gara>.csv`, lo 03 e lo 04 li leggono. Così si può rilanciare solo il passo che interessa.
- Ogni script aggiunge la cartella del progetto a `sys.path`, così `from src.clean import prepara_gara`
  funziona anche lanciando `python scripts/02_pulisci.py` dal terminale (PyCharm lo fa già da solo).
- I percorsi di `data/` e `figures/` sono calcolati da `src/data.py` (`Path(__file__)`), non dalla cartella corrente.
- **Verifica**: ho confrontato l'output dei 4 script con quello del vecchio `main.py`: tutti i numeri sono identici,
  e `data/degrado_per_gara.csv` è uguale byte per byte.
- `requirements.txt` elenca solo le 5 librerie usate dal codice (con le versioni del `.venv`), non tutto il `pip freeze`,
  che conteneva anche Jupyter e FastF1.
