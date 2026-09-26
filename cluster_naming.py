"""
cluster_naming.py
=================
Automatyczne, ODPORNE NA PRZETASOWANIE etykiet nadawanie nazw merytorycznych
klastrom k=5 na podstawie ich profilu cech (a nie numeru, który K-Means losuje
przy każdym uruchomieniu).

Logika oparta na profilach z analizy (heatmapa Z-score, standaryzacja populacyjna):

  • ELITA ZAROBKOWA   – ekstremalnie wysokie WWZ (zarobki) ponad wszystkimi
  • B2B / SAMOZATR.   – najwyższe SAMOZ przy jednocześnie niskim ETAT
  • TRUDNY START      – najwyższy CZAS_PRACA (długie szukanie) + wysokie PROC_STUDIA
  • STABILNY ETAT     – dodatni ETAT, zarobki wokół/nieco powyżej średniej
  • PRZECIĘTNI/RYZYKO – reszta: bladszy profil, podwyższone WWB (bezrobocie)

Funkcja przydziela każdą z 5 etykiet DOKŁADNIE RAZ, w kolejności od
najbardziej jednoznacznej (elita po WWZ) do najmniej (przeciętni = reszta),
więc wynik jest deterministyczny niezależnie od numeracji K-Means.

Nazwa zwracana w formacie "Nazwa merytoryczna (N)", gdzie N = oryginalny
numer klastra — zgodnie z wymaganiem (numer w nawiasie).
"""

import numpy as np
import pandas as pd


def _mean_over(profile_z: pd.DataFrame, substrings) -> pd.Series:
    """Średni Z-score po kolumnach (cechach), których nazwa zawiera dowolny z substringów.
    profile_z: indeks = id klastra, kolumny = cechy (Z-score populacyjny)."""
    cols = [c for c in profile_z.columns
            if any(s in c for s in substrings)]
    if not cols:
        return pd.Series(0.0, index=profile_z.index)
    return profile_z[cols].mean(axis=1)


def assign_cluster_names(df, cluster_col, features, k_expected=5):
    """
    Zwraca:
      name_map : dict {cluster_id -> "Nazwa (id)"}
      label_map: dict {cluster_id -> "Nazwa"}            (bez numeru, do legend)
      report   : pd.DataFrame z osiami profilu użytymi do decyzji (audyt)

    Wymaga, by w df był cluster_col oraz kolumny `features`.
    Standaryzacja populacyjna liczona wewnątrz (spójna z heatmapą profili).
    """
    ids = sorted(df[cluster_col].unique())
    if len(ids) != k_expected:
        # Działa dla dowolnego k, ale nazewnictwo zaprojektowano pod k=5.
        pass

    gmean = df[features].mean()
    gstd = df[features].std().replace(0, np.nan)
    prof = df.groupby(cluster_col)[features].mean()
    prof_z = (prof - gmean) / gstd          # id klastra × cechy

    # --- osie profilu (średnie Z-score po grupach cech) ---
    axes = pd.DataFrame(index=prof_z.index)
    axes['WWZ']   = _mean_over(prof_z, ['WWZ'])                 # zarobki (wyżej = lepiej)
    axes['WWB']   = _mean_over(prof_z, ['WWB'])                 # bezrobocie (wyżej = gorzej)
    axes['ETAT']  = _mean_over(prof_z, ['CZY_ETAT'])           # umowa o pracę
    axes['SAMOZ'] = _mean_over(prof_z, ['CZY_SAMOZ'])          # samozatrudnienie/B2B
    axes['CZAS']  = _mean_over(prof_z, ['CZAS_PRACA'])         # czas do 1. pracy (wyżej = gorzej)
    axes['STUDIA']= _mean_over(prof_z, ['PROC_STUDIA'])        # kontynuacja nauki

    remaining = set(prof_z.index)
    assignment = {}   # cluster_id -> nazwa (bez numeru)

    def take(cluster_id, name):
        assignment[cluster_id] = name
        remaining.discard(cluster_id)

    # 1) ELITA ZAROBKOWA: najwyższe WWZ spośród pozostałych
    elita = axes.loc[list(remaining), 'WWZ'].idxmax()
    take(elita, 'Elita zarobkowa')

    # 2) B2B / SAMOZATRUDNIENIE: najwyższe SAMOZ spośród pozostałych
    b2b = axes.loc[list(remaining), 'SAMOZ'].idxmax()
    take(b2b, 'B2B / samozatrudnienie')

    # 3) TRUDNY START: najwyższy CZAS_PRACA (długie szukanie pracy) spośród pozostałych
    trudny = axes.loc[list(remaining), 'CZAS'].idxmax()
    take(trudny, 'Trudny start')

    # 4) STABILNY ETAT: najwyższe ETAT spośród pozostałych
    if remaining:
        etat = axes.loc[list(remaining), 'ETAT'].idxmax()
        take(etat, 'Stabilny etat')

    # 5) PRZECIĘTNI / RYZYKO BEZROBOCIA: to, co zostało
    for cid in list(remaining):
        take(cid, 'Przeciętni / ryzyko bezrobocia')

    name_map = {cid: f"{assignment[cid]} ({cid})" for cid in ids}
    label_map = {cid: assignment[cid] for cid in ids}

    report = axes.copy()
    report['nazwa'] = [assignment[cid] for cid in report.index]
    return name_map, label_map, report
