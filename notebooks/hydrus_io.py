"""
hydrus_io.py — Lecture des fichiers de sortie de HYDRUS-1D (version 4.x)
=========================================================================
Cours « Physique et hydrodynamique des sols » — S. J. Gumiere

Fonctions
---------
read_tlevel(path)     -> DataFrame indexé par le temps (T_LEVEL.OUT)
read_nod_inf(path)    -> dict {temps: DataFrame(Node, Depth, Head, Moisture, K, C, Flux, ...)}
read_obs_node(path)   -> DataFrame multi-colonnes (OBS_NODE.OUT) : colonnes (noeud, variable)
read_balance(path)    -> DataFrame indexé par le temps (BALANCE.OUT, sous-région globale)
read_solute(path)     -> DataFrame indexé par le temps (solute1.out, solute2.out, ...)
read_run_inf(path)    -> DataFrame (RUN_INF.OUT : pas de temps, itérations)
read_a_level(path)    -> DataFrame (A_LEVEL.OUT : bilans aux pas d'impression atmosphériques)
profiles_to_array(nod)-> (z, times, H, THETA) tableaux 2D pratiques pour tracer

Toutes les fonctions acceptent soit le chemin d'un fichier, soit le chemin d'un
dossier de projet HYDRUS-1D (le nom standard du fichier est alors ajouté).

Exemple
-------
>>> from hydrus_io import read_tlevel, read_nod_inf
>>> tl = read_tlevel("hydrus/J06_infiltration_loam")
>>> tl["sum(Infil)"].plot()
>>> nod = read_nod_inf("hydrus/J06_infiltration_loam")
>>> for t, df in nod.items(): plt.plot(df["Moisture"], df["Depth"], label=f"t = {t}")
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "read_tlevel", "read_nod_inf", "read_obs_node", "read_balance",
    "read_solute", "read_run_inf", "read_a_level", "profiles_to_array",
]


def _resolve(path, default_name: str) -> Path:
    p = Path(path)
    if p.is_dir():
        # recherche insensible à la casse (HYDRUS écrit parfois en minuscules)
        for f in p.iterdir():
            if f.name.lower() == default_name.lower():
                return f
        raise FileNotFoundError(f"{default_name} introuvable dans {p}")
    return p


def _to_float(tok: str) -> float:
    """Convertit un nombre Fortran (gère les '*****' de débordement et les '-0.1E+02')."""
    try:
        return float(tok)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------
# T_LEVEL.OUT
# ----------------------------------------------------------------------
def read_tlevel(path) -> pd.DataFrame:
    """Lit T_LEVEL.OUT : flux et charges aux limites à chaque pas de temps.

    Colonnes principales : rTop (flux potentiel en surface), vTop (flux réel en
    surface), vRoot, vBot, sum(vTop), sum(vBot), hTop, hBot, RunOff, Volume, ...
    Convention HYDRUS : flux négatif = vers le bas (infiltration), positif = vers le haut.
    """
    f = _resolve(path, "T_LEVEL.OUT")
    lines = f.read_text(errors="ignore").splitlines()
    # ligne d'en-tête : celle qui commence par "Time" et contient "rTop"
    ih = next(i for i, l in enumerate(lines) if l.strip().startswith("Time") and "rTop" in l)
    cols = lines[ih].split()
    rows = []
    for l in lines[ih + 2:]:
        s = l.strip()
        if not s or s.lower().startswith("end"):
            continue
        toks = s.split()
        if not re.match(r"^[-+0-9.]", toks[0]):
            continue
        rows.append([_to_float(t) for t in toks])
    n = max(len(r) for r in rows)
    cols = cols[:n] if len(cols) >= n else cols + [f"col{i}" for i in range(len(cols), n)]
    df = pd.DataFrame([r + [np.nan] * (n - len(r)) for r in rows], columns=cols)
    df = df.set_index("Time")
    return df


# ----------------------------------------------------------------------
# NOD_INF.OUT
# ----------------------------------------------------------------------
def read_nod_inf(path) -> dict[float, pd.DataFrame]:
    """Lit NOD_INF.OUT : profils nodaux (h, theta, K, C, flux, ...) aux temps d'impression.

    Retourne un dictionnaire {temps: DataFrame}. Les profondeurs (Depth) sont
    négatives vers le bas (convention HYDRUS, z = 0 en surface).
    """
    f = _resolve(path, "NOD_INF.OUT")
    lines = f.read_text(errors="ignore").splitlines()
    out: dict[float, pd.DataFrame] = {}
    i = 0
    n = len(lines)
    while i < n:
        m = re.match(r"^\s*Time:\s*([-+0-9.Ee]+)", lines[i])
        if m:
            t = float(m.group(1))
            # en-tête de colonnes = première ligne non vide commençant par 'Node'
            j = i + 1
            while j < n and not lines[j].strip().startswith("Node"):
                j += 1
            cols = lines[j].split()
            j += 2  # saute la ligne des unités
            rows = []
            while j < n:
                s = lines[j].strip()
                if not s:
                    j += 1
                    if rows:  # fin du bloc
                        break
                    continue
                if s.lower().startswith("end") or s.startswith("*******") or s.startswith("Time:"):
                    break
                toks = s.split()
                if not toks[0].isdigit():
                    j += 1
                    continue
                rows.append([_to_float(x) for x in toks])
                j += 1
            if rows:
                w = max(len(r) for r in rows)
                c = cols[:w] if len(cols) >= w else cols + [f"col{k}" for k in range(len(cols), w)]
                df = pd.DataFrame([r + [np.nan] * (w - len(r)) for r in rows], columns=c)
                df["Node"] = df["Node"].astype(int)
                out[t] = df.set_index("Node")
            i = j
        else:
            i += 1
    return out


def profiles_to_array(nod: dict[float, pd.DataFrame], var_h="Head", var_th="Moisture"):
    """Convertit le dictionnaire de read_nod_inf en tableaux 2D.

    Retourne z (profondeurs, négatives), times (1D), H[t, z], THETA[t, z].
    """
    times = np.array(sorted(nod.keys()))
    z = nod[times[0]]["Depth"].to_numpy()
    H = np.vstack([nod[t][var_h].to_numpy() for t in times])
    TH = np.vstack([nod[t][var_th].to_numpy() for t in times])
    return z, times, H, TH


# ----------------------------------------------------------------------
# OBS_NODE.OUT
# ----------------------------------------------------------------------
def read_obs_node(path) -> pd.DataFrame:
    """Lit OBS_NODE.OUT : séries temporelles aux nœuds d'observation.

    Retourne un DataFrame à colonnes MultiIndex (numéro de nœud, variable), où
    variable ∈ {h, theta, Temp, Conc, Flux, ...} selon les modules activés.
    """
    f = _resolve(path, "OBS_NODE.OUT")
    lines = f.read_text(errors="ignore").splitlines()
    inode = next(i for i, l in enumerate(lines) if "Node(" in l)
    nodes = [int(x) for x in re.findall(r"Node\(\s*(\d+)\)", lines[inode])]
    ih = next(i for i in range(inode, len(lines)) if lines[i].strip().startswith("time"))
    hdr = lines[ih].split()[1:]  # sans 'time'
    nvar = len(hdr) // len(nodes)
    varnames = hdr[:nvar]
    rows = []
    for l in lines[ih + 1:]:
        s = l.strip()
        if not s or s.lower().startswith("end"):
            continue
        toks = s.split()
        if not re.match(r"^[-+0-9.]", toks[0]):
            continue
        rows.append([_to_float(x) for x in toks])
    arr = np.array(rows)
    t = arr[:, 0]
    cols = pd.MultiIndex.from_tuples([(nd, v) for nd in nodes for v in varnames], names=["node", "var"])
    df = pd.DataFrame(arr[:, 1:1 + len(cols)], index=pd.Index(t, name="Time"), columns=cols)
    return df


# ----------------------------------------------------------------------
# BALANCE.OUT
# ----------------------------------------------------------------------
def read_balance(path) -> pd.DataFrame:
    """Lit BALANCE.OUT (bilan de masse global) : W-volume, In-flow, h Mean,
    Top Flux, Bot Flux, WatBalT (erreur absolue), WatBalR (erreur relative, %).
    Si plusieurs sous-régions existent, seule la première colonne (totale) est lue.
    """
    f = _resolve(path, "BALANCE.OUT")
    txt = f.read_text(errors="ignore")
    blocks = re.split(r"\n\s*Time\s+\[T\]\s+", txt)[1:]
    recs = []
    for b in blocks:
        t = float(b.split()[0])
        rec = {"Time": t}
        for key in ["Area", "W-volume", "In-flow", "h Mean", "Top Flux", "Bot Flux",
                    "WatBalT", "WatBalR", "ConcVol", "ConcVolIm", "cMean", "CncBalT", "CncBalR",
                    "cMeanIm", "Volume", "TopFlux", "BotFlux"]:
            m = re.search(r"^\s*" + re.escape(key) + r"\s*(?:\[[^\]]*\])?\s+([-+0-9.Ee*]+)", b, re.M)
            if m:
                rec[key] = _to_float(m.group(1))
        recs.append(rec)
    return pd.DataFrame(recs).set_index("Time")


# ----------------------------------------------------------------------
# solute1.out, ...
# ----------------------------------------------------------------------
def read_solute(path, k: int = 1) -> pd.DataFrame:
    """Lit solute{k}.out : flux de soluté aux limites (cvTop, cvBot, sum(cvTop),
    sum(cvBot), cTop, cBot, ...). Concentrations en M L^-3, flux en M L^-2 T^-1."""
    f = _resolve(path, f"solute{k}.out")
    lines = f.read_text(errors="ignore").splitlines()
    ih = next(i for i, l in enumerate(lines) if l.strip().startswith("Time") and "cvTop" in l)
    cols = lines[ih].split()
    rows = []
    for l in lines[ih + 2:]:
        s = l.strip()
        if not s or s.lower().startswith("end"):
            continue
        toks = s.split()
        if not re.match(r"^[-+0-9.]", toks[0]):
            continue
        rows.append([_to_float(x) for x in toks])
    n = max(len(r) for r in rows)
    cols = cols[:n] if len(cols) >= n else cols + [f"col{i}" for i in range(len(cols), n)]
    df = pd.DataFrame([r + [np.nan] * (n - len(r)) for r in rows], columns=cols)
    return df.set_index("Time")


# ----------------------------------------------------------------------
# RUN_INF.OUT
# ----------------------------------------------------------------------
def read_run_inf(path) -> pd.DataFrame:
    """Lit RUN_INF.OUT : informations sur les pas de temps (dt, itérations, convergence)."""
    f = _resolve(path, "RUN_INF.OUT")
    lines = f.read_text(errors="ignore").splitlines()
    ih = next(i for i, l in enumerate(lines) if "TLevel" in l and "Iter" in l)
    cols = lines[ih].split()
    rows = []
    for l in lines[ih + 1:]:
        s = l.strip()
        if not s or s.lower().startswith("end"):
            continue
        toks = s.split()
        if not toks[0].isdigit():
            continue
        vals = [_to_float(x) if x not in ("T", "F") else (x == "T") for x in toks]
        rows.append(vals)
    n = max(len(r) for r in rows)
    return pd.DataFrame([r + [np.nan] * (n - len(r)) for r in rows], columns=cols[:n])


# ----------------------------------------------------------------------
# A_LEVEL.OUT
# ----------------------------------------------------------------------
def read_a_level(path) -> pd.DataFrame:
    """Lit A_LEVEL.OUT : flux cumulés aux pas de temps atmosphériques."""
    f = _resolve(path, "A_LEVEL.OUT")
    lines = f.read_text(errors="ignore").splitlines()
    ih = next(i for i, l in enumerate(lines) if l.strip().startswith("Time") and "sum(rTop)" in l)
    cols = lines[ih].split()
    rows = []
    for l in lines[ih + 2:]:
        s = l.strip()
        if not s or s.lower().startswith("end"):
            continue
        toks = s.split()
        if not re.match(r"^[-+0-9.]", toks[0]):
            continue
        rows.append([_to_float(x) for x in toks])
    n = max(len(r) for r in rows)
    cols = cols[:n] if len(cols) >= n else cols + [f"col{i}" for i in range(len(cols), n)]
    df = pd.DataFrame([r + [np.nan] * (n - len(r)) for r in rows], columns=cols)
    return df.set_index("Time")


if __name__ == "__main__":  # petit auto-test
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    tl = read_tlevel(d)
    print("T_LEVEL :", tl.shape, list(tl.columns)[:6])
    nod = read_nod_inf(d)
    print("NOD_INF :", len(nod), "profils ; premier temps =", min(nod))
    try:
        ob = read_obs_node(d)
        print("OBS_NODE :", ob.shape, ob.columns.get_level_values(0).unique().tolist())
    except Exception as e:
        print("OBS_NODE :", e)
    bal = read_balance(d)
    print("BALANCE :", bal.shape, "WatBalR max =", bal.get("WatBalR", pd.Series([np.nan])).abs().max())
