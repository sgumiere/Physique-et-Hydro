"""
generate_hydrus_runs.py — Génération des projets HYDRUS-1D de référence du cours
================================================================================
Ce script (annexe instructeur) construit et exécute, avec le paquet Python
`phydrus` et l'exécutable HYDRUS-1D compilé sous Linux/macOS, toutes les
simulations utilisées dans les exercices des jours 5 à 10. Les étudiants
construisent les mêmes projets dans l'interface graphique HYDRUS-1D (Windows)
en suivant les consignes des diapositives ; les fichiers produits ici servent
de référence (entrées SELECTOR.IN / PROFILE.DAT / ATMOSPH.IN et sorties
T_LEVEL.OUT / NOD_INF.OUT / OBS_NODE.OUT / BALANCE.OUT / solute1.out).

Unités : L = cm, T = jour, M = g (masse de soluté en mg pour les nitrates).
Convention HYDRUS : flux positif vers le haut ; l'infiltration est négative.

Usage : python generate_hydrus_runs.py [--exe /chemin/vers/hydrus] [--only J06]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import phydrus as ps

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Paramètres de van Genuchten–Mualem (Carsel & Parrish, 1988) — cm, jour
#            thr    ths    alpha  n     Ks (cm/j)  l
SOLS = {
    "sable":           [0.045, 0.430, 0.145, 2.68, 712.8, 0.5],
    "sable_limoneux":  [0.057, 0.410, 0.124, 2.28, 350.2, 0.5],
    "loam_sableux":    [0.065, 0.410, 0.075, 1.89, 106.1, 0.5],
    "loam":            [0.078, 0.430, 0.036, 1.56, 24.96, 0.5],
    "loam_limoneux":   [0.067, 0.450, 0.020, 1.41, 10.80, 0.5],
    "loam_argileux":   [0.095, 0.410, 0.019, 1.31, 6.24, 0.5],
    "argile":          [0.068, 0.380, 0.008, 1.09, 4.80, 0.5],
}


def vg_theta(h, p):
    thr, ths, a, n = p[:4]
    m = 1 - 1 / n
    h = np.asarray(h, float)
    Se = np.where(h < 0, (1 + (a * np.abs(h)) ** n) ** (-m), 1.0)
    return thr + (ths - thr) * Se


def vg_K(h, p):
    thr, ths, a, n, Ks, l = p
    m = 1 - 1 / n
    h = np.asarray(h, float)
    Se = np.where(h < 0, (1 + (a * np.abs(h)) ** n) ** (-m), 1.0)
    return Ks * Se ** l * (1 - (1 - Se ** (1 / m)) ** m) ** 2


def h_for_K(q, p):
    """Charge de pression h (<0) telle que K(h) = q (régime permanent à gradient unitaire)."""
    from scipy.optimize import brentq
    return brentq(lambda h: vg_K(h, p) - q, -1e5, -1e-6)


# ----------------------------------------------------------------------------
def new_model(exe, name, desc, tmax, dt=1e-3, dtmin=1e-7, dtmax=None, dtprint=None,
              printinit=None, printmax=None, mass_units="g"):
    ws = HERE / name
    if ws.exists():
        shutil.rmtree(ws)
    ml = ps.Model(exe_name=exe, ws_name=str(ws), name="model", description=desc,
                  mass_units=mass_units, time_unit="days", length_unit="cm")
    ml.basic_info["lShort"] = False      # sorties T_LEVEL / solute à chaque pas de temps
    dtmax = dtmax if dtmax is not None else tmax / 20
    dtprint = dtprint if dtprint is not None else tmax / 20
    ml.add_time_info(tinit=0, tmax=tmax, print_times=True, dt=dt, dtmin=dtmin, dtmax=dtmax,
                     printinit=printinit if printinit is not None else dtprint,
                     printmax=(printmax if printmax is not None else tmax) + 1e-9, dtprint=dtprint)
    return ml


def materials(ml, *noms, override=None):
    m = ml.get_empty_material_df(n=len(noms))
    for i, nm in enumerate(noms, start=1):
        p = list(override[nm]) if (override and nm in override) else list(SOLS[nm])
        m.loc[i, ("water", slice(None))] = p if isinstance(m.columns, pd.MultiIndex) else p
    ml.add_material(m)
    return m


def run(ml, readme: str):
    ml.write_input()
    ws = Path(ml.ws_name)
    # phydrus laisse parfois une ligne vide après la liste TPrint : HYDRUS la lit comme un
    # enregistrement et décale la lecture des blocs suivants -> on supprime les lignes vides.
    sel = ws / "SELECTOR.IN"
    lines = [l for l in sel.read_text().splitlines() if l.strip()]
    # Bloc G : HYDRUS lit « iMoSink, cRootMax(1..NS), OmegaC » ; sans soluté (NS = 0) phydrus écrit tout de même
    # cRootMax, si bien qu'HYDRUS lisait OmegaC = 0 (compensation totale !). On retire cRootMax quand NS = 0.
    if ml.solutes is None or len(ml.solutes) == 0:
        for i, l in enumerate(lines):
            if l.strip().startswith("iMoSink") and "cRootMax" in l:
                toks = lines[i + 1].split()
                lines[i] = "iMoSink OmegaC"
                lines[i + 1] = f"{toks[0]} {toks[-1]}"
    sel.write_text("\n".join(lines) + "\n")
    r = ml.simulate()
    (ws / "README.txt").write_text(readme.strip() + "\n", encoding="utf-8")
    # nettoyage des fichiers volumineux inutiles
    for f in ["I_CHECK.OUT", "PROFILE.OUT"]:
        if (ws / f).exists() and (ws / f).stat().st_size > 2e6:
            (ws / f).unlink()
    err = (ws / "Error.msg")
    ok = r.returncode == 0 and (ws / "T_LEVEL.OUT").stat().st_size > 1000 and not err.exists()
    if err.exists():
        print("   Error.msg :", err.read_text(errors="ignore").strip()[:200])
    print(f"[{'OK' if ok else 'ECHEC'}] {ws.name}  —  {readme.strip().splitlines()[0]}")
    return ok


# ============================================================================
# JOUR 5 — régime permanent : colonne avec nappe
# ============================================================================
def j05(exe):
    """Colonne de loam de 100 cm, nappe au fond (h = 0), flux constant en surface.
    Deux cas : évaporation stationnaire (q = +0.03 cm/j, sous le maximum ≈ 0.054 cm/j) et infiltration (q = -1 cm/j)."""
    for tag, q in [("evaporation", 0.03), ("infiltration", -1.0)]:
        name = f"J05_permanent_nappe_{tag}"
        ml = new_model(exe, name, f"Regime permanent loam, nappe a -100 cm, q_top = {q} cm/j",
                       tmax=200, dt=0.01, dtmax=5, dtprint=20)
        # Haut : flux constant (code 1) ; bas : charge constante h = 0 (nappe)
        ml.add_waterflow(model=0, top_bc=1, bot_bc=0, rtop=q, rbot=0, maxit=20, tolth=1e-4, tolh=0.1, rroot=0)
        materials(ml, "loam")
        prof = ps.create_profile(top=0, bot=-100, dx=1, h=0)
        prof["h"] = -100.0 - prof["x"]          # équilibre hydrostatique initial : H = -100 cm
        prof.loc[prof.index[-1], "h"] = 0.0
        ml.add_profile(prof)
        ml.add_obs_nodes([-10, -30, -50, -80])
        run(ml, f"""
{name} : régime permanent, loam (Carsel & Parrish), colonne 0–100 cm
Interface HYDRUS-1D :
  Main processes : Water flow. Time units : days ; length : cm. Final time 200 d.
  Water flow parameters : loam (thr 0.078, ths 0.43, alpha 0.036, n 1.56, Ks 24.96, l 0.5).
  Boundary conditions : Upper = Constant flux ({q} cm/j, positif = évaporation) ;
                        Lower = Constant pressure head (h = 0 : nappe au bas de la colonne).
  Profile : 101 noeuds (dz = 1 cm), condition initiale hydrostatique h(z) = -100 - z (H = -100 cm).
  Observation nodes : 10, 30, 50, 80 cm. Print times : tous les 20 jours.
Résultat attendu : profil h(z) stationnaire à t = 200 j, à comparer avec l'intégration
de dh/dz = -1 - q/K(h) (Buckingham–Darcy en régime permanent).
""")


# ============================================================================
# JOUR 6 — infiltration submergée et pluie à flux imposé
# ============================================================================
def j06(exe):
    name = "J06_infiltration_submergee_loam"
    ml = new_model(exe, name, "Infiltration submergee (h=0 en surface), loam, h_init=-100 cm",
                   tmax=1.0, dt=1e-4, dtmax=0.02, dtprint=0.05)
    ml.add_waterflow(model=0, top_bc=0, bot_bc=4, rtop=0, maxit=20, tolth=1e-4, tolh=0.1, rroot=0)
    materials(ml, "loam")
    prof = ps.create_profile(top=0, bot=-100, dx=1, h=-100)
    prof.loc[prof.index[0], "h"] = 0.0
    ml.add_profile(prof)
    ml.add_obs_nodes([-10, -30, -50])
    run(ml, f"""
{name} : infiltration submergée dans un loam (HYDRUS Simulation 5.1 de Radcliffe & Šimůnek)
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d ; initial time step 1e-4 d ; max 0.02 d.
  Water flow parameters : loam (0.078, 0.43, 0.036, 1.56, 24.96, 0.5).
  Boundary conditions : Upper = Constant pressure head (h = 0 : lame d'eau nulle) ;
                        Lower = Free drainage.
  Profile : 101 noeuds, dz = 1 cm ; h initial = -100 cm partout (h = 0 au noeud de surface).
  Observation nodes : 10, 30, 50 cm. Print times : 0.05, 0.10, ..., 1.0 d.
Résultat attendu : front d'humectation atteignant le bas vers t ≈ 0.6 d ; flux en surface
tendant vers Ks = 24.96 cm/j ; infiltration cumulée ≈ 25.8 cm à 1 d.
""")

    name = "J06_pluie_flux_impose_loam"
    ml = new_model(exe, name, "Pluie 30 cm/j pendant 0.5 j (> Ks) puis 0 : apparition du ruissellement",
                   tmax=1.0, dt=1e-4, dtmax=0.01, dtprint=0.05)
    ml.add_waterflow(model=0, top_bc=3, bot_bc=4, maxit=20, tolth=1e-4, tolh=0.1, rroot=0)
    materials(ml, "loam")
    prof = ps.create_profile(top=0, bot=-100, dx=1, h=-300)
    ml.add_profile(prof)
    ml.add_obs_nodes([-5, -20, -50])
    atm = pd.DataFrame({"tAtm": [0.5, 1.0], "Prec": [30.0, 0.0], "rSoil": [0.0, 0.0],
                        "rRoot": [0.0, 0.0], "hCritA": [1e5, 1e5]})
    ml.add_atmospheric_bc(atm, hcrits=0.0, tatm=0.0, prec=0.0, rsoil=0.0, rroot=0.0, hcrita=1e5, rb=0.0, hb=0.0, ht=0.0, ttop=0.0, tbot=0.0, ampl=0.0)
    run(ml, f"""
{name} : pluie de 30 cm/j (> Ks = 24.96 cm/j) pendant 0.5 j, puis arrêt
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d.
  Boundary conditions : Upper = Atmospheric BC with surface run off (hCritS = 0) ;
                        Lower = Free drainage.
  Time-variable BC (ATMOSPH.IN) : 2 enregistrements : t = 0.5 (Prec = 30), t = 1.0 (Prec = 0).
  Profile : 101 noeuds, h initial = -300 cm. Observation nodes 5, 20, 50 cm.
Résultat attendu : vTop = rTop tant que la capacité d'infiltration > 30 cm/j ; ensuite h_top = 0,
vTop < rTop et RunOff > 0 (T_LEVEL.OUT). Comparer le temps de submersion à Green–Ampt.
""")


# ============================================================================
# JOUR 7 — sols stratifiés et croûte de surface
# ============================================================================
def j07(exe):
    for tag, haut, bas in [("loam_sur_sable", "loam", "sable"), ("sable_sur_loam", "sable", "loam")]:
        name = f"J07_stratifie_{tag}"
        ml = new_model(exe, name, f"Infiltration submergee, {haut} 0-30 cm sur {bas} 30-100 cm",
                       tmax=1.0, dt=1e-5, dtmin=1e-8, dtmax=0.01, dtprint=0.05)
        ml.add_waterflow(model=0, top_bc=0, bot_bc=4, rtop=0, maxit=30, tolth=1e-4, tolh=0.1, rroot=0)
        materials(ml, haut, bas)
        prof = ps.create_profile(top=0, bot=-100, dx=1, h=-200)
        prof.loc[prof["x"] < -30, "Mat"] = 2
        prof.loc[prof["x"] < -30, "Lay"] = 2
        prof.loc[prof.index[0], "h"] = 0.0
        ml.add_profile(prof)
        ml.add_obs_nodes([-10, -25, -35, -60])
        run(ml, f"""
{name} : infiltration submergée dans un profil à deux couches ({haut} 0–30 cm / {bas} 30–100 cm)
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d ; dt initial 1e-5 d, dt max 0.01 d.
  Water flow parameters : 2 matériaux (Carsel & Parrish) : 1 = {haut}, 2 = {bas}.
  Boundary conditions : Upper = Constant pressure head (h = 0) ; Lower = Free drainage.
  Profile : 101 noeuds ; matériau 1 de 0 à -30 cm, matériau 2 en dessous ; h initial = -200 cm.
  Observation nodes : 10, 25, 35, 60 cm.
Résultat attendu : ralentissement du front à l'interface (barrière capillaire ou limitation
par la couche fine) ; comparer les flux vTop et les profils h(z), theta(z) des deux cas.
""")

    for tag, crust in [("sans_croute", False), ("avec_croute", True)]:
        name = f"J07_croute_{tag}"
        ml = new_model(exe, name, "Pluie 10 cm/j pendant 0.5 j sur un loam " + ("avec croûte de 1 cm (Ks = 0.5 cm/j)" if crust else "sans croûte"),
                       tmax=1.0, dt=1e-5, dtmin=1e-8, dtmax=0.01, dtprint=0.05)
        ml.add_waterflow(model=0, top_bc=3, bot_bc=4, maxit=30, tolth=1e-4, tolh=0.1, rroot=0)
        if crust:
            croute = [0.078, 0.43, 0.036, 1.56, 0.5, 0.5]   # même rétention, Ks réduit
            materials(ml, "loam_croute", "loam", override={"loam_croute": croute})
        else:
            materials(ml, "loam")
        prof = ps.create_profile(top=0, bot=-100, dx=0.5 if crust else 1, h=-300)
        if crust:
            prof.loc[prof["x"] < -1.0, "Mat"] = 2
            prof.loc[prof["x"] < -1.0, "Lay"] = 2
            prof.loc[prof["x"] >= -1.0, "Mat"] = 1
        ml.add_profile(prof)
        ml.add_obs_nodes([-2, -10, -30])
        atm = pd.DataFrame({"tAtm": [0.5, 1.0], "Prec": [10.0, 0.0], "rSoil": [0.0, 0.0],
                            "rRoot": [0.0, 0.0], "hCritA": [1e5, 1e5]})
        ml.add_atmospheric_bc(atm, hcrits=0.0, tatm=0.0, prec=0.0, rsoil=0.0, rroot=0.0, hcrita=1e5, rb=0.0, hb=0.0, ht=0.0, ttop=0.0, tbot=0.0, ampl=0.0)
        run(ml, f"""
{name} : pluie de 10 cm/j pendant 0.5 j sur un loam {'avec' if crust else 'sans'} croûte de battance
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d.
  Water flow parameters : {'2 matériaux : 1 = croûte (loam avec Ks = 0.5 cm/j), 2 = loam' if crust else '1 matériau : loam'}.
  Boundary conditions : Upper = Atmospheric BC with surface run off (hCritS = 0) ; Lower = Free drainage.
  ATMOSPH.IN : t = 0.5 (Prec = 10), t = 1.0 (Prec = 0).
  Profile : {'201 noeuds (dz = 0.5 cm), croûte de 0 à -1 cm' if crust else '101 noeuds (dz = 1 cm)'} ; h initial = -300 cm.
Résultat attendu : {'ruissellement important dès les premières minutes, infiltration limitée par la croûte' if crust else 'aucun ruissellement (10 cm/j < Ks), toute la pluie s infiltre'}.
""")


# ============================================================================
# JOUR 8 — redistribution et saison de culture
# ============================================================================
def meteo_saison(seed=42, ndays=120):
    rng = np.random.default_rng(seed)
    t = np.arange(1, ndays + 1)
    # ET0 saisonnière (mm/j) : 2.5 -> 5.5 -> 3
    et0 = 4.0 + 1.6 * np.sin(2 * np.pi * (t - 20) / 240) + rng.normal(0, 0.4, ndays)
    et0 = np.clip(et0, 1.0, 7.0)
    # pluie : événements aléatoires (mm/j)
    pluie = np.where(rng.random(ndays) < 0.28, rng.gamma(1.6, 6.0, ndays), 0.0)
    pluie = np.round(pluie, 1)
    # coefficient cultural (maïs) : 0.4 (0-20 j) -> 1.15 (55-90 j) -> 0.6 (120 j)
    kc = np.interp(t, [1, 20, 55, 90, 120], [0.40, 0.40, 1.15, 1.15, 0.60])
    lai = np.interp(t, [1, 20, 55, 90, 120], [0.2, 0.5, 3.5, 4.0, 2.0])
    etc = kc * et0
    ep = etc * np.exp(-0.463 * lai)           # évaporation potentielle du sol
    tp = etc - ep                              # transpiration potentielle
    return pd.DataFrame({"jour": t, "pluie_mm": pluie, "ET0_mm": np.round(et0, 2),
                         "Kc": np.round(kc, 3), "LAI": np.round(lai, 2),
                         "ETc_mm": np.round(etc, 2), "Ep_mm": np.round(ep, 2), "Tp_mm": np.round(tp, 2)})


def j08(exe):
    name = "J08_redistribution_loam"
    ml = new_model(exe, name, "Drainage interne d'un profil initialement proche de la saturation (h = -10 cm), sans flux en surface",
                   tmax=30, dt=1e-4, dtmax=0.5, dtprint=1.0)
    ml.add_waterflow(model=0, top_bc=1, bot_bc=4, rtop=0.0, maxit=20, tolth=1e-4, tolh=0.1, rroot=0)
    materials(ml, "loam")
    prof = ps.create_profile(top=0, bot=-100, dx=1, h=-10)
    ml.add_profile(prof)
    ml.add_obs_nodes([-10, -25, -50, -90])
    run(ml, f"""
{name} : drainage interne (« méthode du profil couvert ») d'un loam de 100 cm, 30 j
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 30 d ; print times chaque jour.
  Boundary conditions : Upper = Constant flux = 0 (surface couverte : ni pluie ni évaporation) ;
                        Lower = Free drainage.
  Profile : 101 noeuds (0 à -100 cm) ; h initial = -10 cm partout (theta ≈ 0.41, proche de la saturation).
  Observation nodes : 10, 25, 50, 90 cm.
Résultat attendu : flux de drainage vBot décroissant rapidement (loi de puissance), teneur en eau
tendant vers la « capacité au champ » dynamique ; comparer theta(t) à 25 cm avec theta(h = -100 cm) et
theta(h = -330 cm) ; stock (Volume) et drainage cumulé sum(vBot) dans T_LEVEL.OUT.
""")

    name = "J08_saison_culture_loam"
    met = meteo_saison()
    met.to_csv(HERE / "J08_meteo_saison.csv", index=False)
    ml = new_model(exe, name, "Saison de culture (mais) 120 j : pluie, evaporation, transpiration, Feddes",
                   tmax=120, dt=1e-3, dtmin=1e-7, dtmax=0.25, dtprint=5.0)
    ml.add_waterflow(model=0, top_bc=3, bot_bc=4, maxit=30, tolth=1e-4, tolh=0.1, rroot=0)
    materials(ml, "loam")
    prof = ps.create_profile(top=0, bot=-150, dx=1, h=-200)
    # distribution racinaire linéaire décroissante jusqu'à 60 cm
    z = prof["x"].to_numpy()
    beta = np.clip(1 + z / 60.0, 0, None)
    prof["Beta"] = beta / beta.sum()
    ml.add_profile(prof)
    ml.add_obs_nodes([-10, -30, -60, -100])
    ml.add_root_uptake(model=0, omegac=1.0, p0=-15, p2h=-325, p2l=-600, p3=-8000, r2h=0.5, r2l=0.1, poptm=[-30])   # omegac = 1 : pas de compensation
    atm = pd.DataFrame({"tAtm": met["jour"].astype(float), "Prec": met["pluie_mm"] / 10.0,
                        "rSoil": met["Ep_mm"] / 10.0, "rRoot": met["Tp_mm"] / 10.0,
                        "hCritA": 15000.0})
    ml.add_atmospheric_bc(atm, hcrits=0.0, tatm=0.0, prec=0.0, rsoil=0.0, rroot=0.0, hcrita=1e5, rb=0.0, hb=0.0, ht=0.0, ttop=0.0, tbot=0.0, ampl=0.0)
    run(ml, f"""
{name} : saison de culture de maïs (120 j) sur un loam, avec prélèvement racinaire (Feddes)
Interface HYDRUS-1D :
  Main processes : Water flow + Root water uptake. Final time 120 d ; print times tous les 5 j.
  Boundary conditions : Upper = Atmospheric BC with surface run off (hCritS = 0) ;
                        Lower = Free drainage. hCritA = 15000 cm.
  Time-variable BC : 120 enregistrements journaliers (fichier J08_meteo_saison.csv, en mm -> cm) :
                     Prec = pluie, rSoil = Ep (évaporation potentielle), rRoot = Tp (transpiration potentielle).
  Root water uptake : Feddes, maïs : P0 = -15, POpt = -30, P2H = -325, P2L = -600, P3 = -8000 cm ; r2H 0.5, r2L 0.1 cm/j ;
                     critical stress index OmegaC = 1 (pas de compensation du prélèvement).
  Root distribution : linéaire décroissante de la surface à 60 cm.
  Profile : 151 noeuds ; h initial = -200 cm. Observation nodes : 10, 30, 60, 100 cm.
Résultat attendu : bilan hydrique (sum(Infil), sum(Evap), sum(vRoot) = transpiration réelle,
sum(vBot) = drainage, Volume = stock) dans T_LEVEL.OUT ; stress hydrique quand vRoot < rRoot.
""")


# ============================================================================
# JOUR 9 — transfert de chaleur
# ============================================================================
def heat_params_loam(ml):
    hp = ml.get_empty_heat_df()
    # Chung & Horton (1987) pour un loam, unités cm, jour, g : b1, b2, b3 en g cm d^-3 K^-1
    # Cn (solides) 1.43327e17, C0 (organique) 1.8737e17, Cw (eau) 3.12035e17 en g cm^-1 d^-2 K^-1
    for i in hp.index:
        hp.loc[i] = [0.57, 0.0, 5.0, 1.56728e19, 2.53474e19, 9.89388e19, 1.43327e17, 1.8737e17, 3.12035e17]
    return hp


def j09(exe):
    # --- sans convection : colonne à teneur en eau uniforme, flux d'eau nul
    name = "J09_chaleur_sans_convection"
    ml = new_model(exe, name, "Onde thermique journaliere sans convection, loam h=-100 cm",
                   tmax=5.0, dt=1e-3, dtmax=0.01, dtprint=0.125)
    ml.add_waterflow(model=0, top_bc=1, bot_bc=1, rtop=0.0, rbot=0.0, maxit=20, rroot=0)
    materials(ml, "loam")
    prof = ps.create_profile(top=0, bot=-100, dx=1, h=-100, temp=20.0)
    ml.add_profile(prof)
    ml.add_obs_nodes([-5, -10, -20, -40])
    ml.add_heat_transport(heat_params_loam(ml), ampl=10.0, top_bc=1, top_temp=20.0, bot_bc=0,
                          bot_temp=20.0, tperiod=1.0, icampbell=0)
    run(ml, f"""
{name} : onde de température journalière (20 ± 10 °C) sans convection dans un loam
Interface HYDRUS-1D :
  Main processes : Water flow + Heat transport. Final time 5 d ; print times toutes les 3 h.
  Water flow : loam ; Upper = Constant flux 0 ; Lower = Constant flux 0 ; h initial = -100 cm (theta ≈ 0.243).
  Heat transport parameters : Chung & Horton, loam (b1 1.56728e19, b2 2.53474e19, b3 9.89388e19 ;
      Cn 1.43327e17, C0 1.8737e17, Cw 3.12035e17 ; fraction solide 0.57 ; dispersivité thermique 5 cm).
  Heat BC : Upper = Temperature BC (Dirichlet) : moyenne 20 °C, amplitude 10 °C, période 1 j ;
            Lower = Zero gradient. Température initiale 20 °C.
  Observation nodes : 5, 10, 20, 40 cm.
Résultat attendu : amortissement exponentiel de l'amplitude et déphasage croissant avec la profondeur ;
comparer à la solution analytique T(z,t) = Tm + A exp(-z/d) sin(omega t - z/d), d = sqrt(2 D_T/omega).
""")

    # --- avec convection : infiltration permanente 5 cm/j, eau entrante à la température de surface
    name = "J09_chaleur_avec_convection"
    q = -5.0
    ml = new_model(exe, name, "Onde thermique journaliere avec infiltration permanente 5 cm/j (convection)",
                   tmax=5.0, dt=1e-3, dtmax=0.01, dtprint=0.125)
    ml.add_waterflow(model=0, top_bc=1, bot_bc=4, rtop=q, maxit=20, rroot=0)
    materials(ml, "loam")
    h0 = h_for_K(-q, SOLS["loam"])
    prof = ps.create_profile(top=0, bot=-100, dx=1, h=h0, temp=20.0)
    ml.add_profile(prof)
    ml.add_obs_nodes([-5, -10, -20, -40])
    ml.add_heat_transport(heat_params_loam(ml), ampl=10.0, top_bc=1, top_temp=20.0, bot_bc=0,
                          bot_temp=20.0, tperiod=1.0, icampbell=0)
    run(ml, f"""
{name} : même onde thermique avec infiltration permanente de 5 cm/j (transport convectif)
Interface HYDRUS-1D :
  Water flow : Upper = Constant flux -5 cm/j ; Lower = Free drainage ; h initial = {h0:.1f} cm (K(h) = 5 cm/j).
  Heat BC : Upper = Temperature BC (Dirichlet, 20 ± 10 °C, période 1 j — l'eau infiltrée entre à la
            température de surface) ; Lower = Zero gradient.
Résultat attendu : pénétration plus profonde de l'onde (convection + conduction) ; comparer les
amplitudes à 20 et 40 cm avec le cas sans convection.
""")


# ============================================================================
# JOUR 10 — transport de solutés
# ============================================================================
def j10(exe):
    p = SOLS["loam_sableux"]
    q = -10.0                                  # cm/j (vers le bas)
    h0 = h_for_K(-q, p)
    th0 = float(vg_theta(h0, p))
    for disp in [0.5, 2.0, 10.0]:
        name = f"J10_traceur_disp{disp:g}cm"
        ml = new_model(exe, name, f"Traceur inerte, loam sableux, q=10 cm/j, dispersivite {disp} cm",
                       tmax=6.0, dt=1e-3, dtmax=0.02, dtprint=0.5)
        ml.add_waterflow(model=0, top_bc=1, bot_bc=4, rtop=q, maxit=20, rroot=0)
        ml.add_solute_transport(model=0, epsi=0.5, lupw=False, lartd=False, top_bc=-1, bot_bc=0, tpulse=0.5)
        m = ml.get_empty_material_df(n=1)
        m.loc[1, ("water", slice(None))] = p
        m.loc[1, ("solute", slice(None))] = [1.5, disp, 1.0, 0.0]   # rho_b, DisperL, Frac, mobile WC
        ml.add_material(m)
        prof = ps.create_profile(top=0, bot=-60, dx=0.5, h=h0, conc=0.0)
        ml.add_profile(prof)
        ml.add_obs_nodes([-10, -20, -40, -60])
        sol = ml.get_empty_solute_df()
        sol.loc[1] = [0.0] * len(sol.columns)     # Kd = 0, pas de dégradation
        sol.loc[1, "beta"] = 1.0                  # exposant de Freundlich = 1 (isotherme linéaire)
        ml.add_solute(sol, difw=1.0, difg=0.0, top_conc=1.0, bot_conc=0.0)
        run(ml, f"""
{name} : traceur inerte en régime permanent non saturé (loam sableux, q = 10 cm/j, θ = {th0:.3f})
Interface HYDRUS-1D :
  Main processes : Water flow + Solute transport (equilibrium). Final time 6 d ; print times 0.5 j.
  Water flow : loam sableux (0.065, 0.41, 0.075, 1.89, 106.1, 0.5) ; Upper = Constant flux -10 cm/j ;
               Lower = Free drainage ; h initial = {h0:.2f} cm (régime permanent, K(h) = 10 cm/j).
  Solute transport : Crank–Nicolson (epsi 0.5), Galerkin ; bulk density 1.5 g/cm3, disp. longitudinale {disp} cm,
               Dw = 1 cm2/j ; Kd = 0, pas de dégradation. Upper = Concentration flux BC, c = 1 pendant
               tPulse = 0.5 j puis 0 ; Lower = Zero concentration gradient.
  Profile : 121 noeuds (dz = 0.5 cm), 0 à -60 cm ; concentration initiale 0. Observation nodes 10, 20, 40, 60 cm.
Résultat attendu : courbe de percée à 60 cm (solute1.out : cvBot ; OBS_NODE.OUT : Conc) — comparer
à la solution analytique de l'ADE pour une injection brève (v = q/θ = {-q/th0:.2f} cm/j, D = {disp}·v + θ·Dw·tau).
""")

    # --- nitrate réactif : sorption linéaire + dégradation du 1er ordre
    name = "J10_nitrate_reactif"
    ml = new_model(exe, name, "Solute reactif (Kd = 0.2 cm3/g, mu = 0.05 /j), loam sableux, q=10 cm/j",
                   tmax=6.0, dt=1e-3, dtmax=0.02, dtprint=0.5)
    ml.add_waterflow(model=0, top_bc=1, bot_bc=4, rtop=q, maxit=20, rroot=0)
    ml.add_solute_transport(model=0, epsi=0.5, lupw=False, lartd=False, top_bc=-1, bot_bc=0, tpulse=0.5)
    m = ml.get_empty_material_df(n=1)
    m.loc[1, ("water", slice(None))] = p
    m.loc[1, ("solute", slice(None))] = [1.5, 2.0, 1.0, 0.0]
    ml.add_material(m)
    prof = ps.create_profile(top=0, bot=-60, dx=0.5, h=h0, conc=0.0)
    ml.add_profile(prof)
    ml.add_obs_nodes([-10, -20, -40, -60])
    sol = ml.get_empty_solute_df()
    sol.loc[1] = [0.0] * len(sol.columns)
    sol.loc[1, "beta"] = 1.0      # isotherme linéaire (exposant de Freundlich = 1)
    sol.loc[1, "ks"] = 0.2        # Kd (cm3/g) -> R = 1 + rho_b Kd / theta
    sol.loc[1, "mu_lw"] = 0.05    # dégradation 1er ordre en phase liquide (1/j)
    sol.loc[1, "mu_sw"] = 0.05    # ... et en phase adsorbée
    ml.add_solute(sol, difw=1.0, difg=0.0, top_conc=1.0, bot_conc=0.0)
    run(ml, f"""
{name} : soluté réactif (sorption linéaire Kd = 0.2 cm3/g, dégradation mu = 0.05 /j) — mêmes conditions
que J10_traceur_disp2cm (loam sableux, q = 10 cm/j, dispersivité 2 cm, pulse de 0.5 j).
Interface HYDRUS-1D : Solute transport parameters : Bulk.d 1.5, Disp. 2, Frac 1 ; Kd = 0.2 ; SnkL1 = SnkS1 = 0.05.
Résultat attendu : facteur de retard R = 1 + rho_b Kd/θ ≈ {1 + 1.5*0.2/th0:.2f} ; pic retardé et atténué par la
dégradation (exp(-mu t)). Comparer la masse récupérée sum(cvBot) au traceur.
""")


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.environ.get("HYDRUS_EXE", str(HERE / "hydrus")))
    ap.add_argument("--only", default=None, help="ex. J06")
    a = ap.parse_args()
    jobs = {"J05": j05, "J06": j06, "J07": j07, "J08": j08, "J09": j09, "J10": j10}
    for k, fn in jobs.items():
        if a.only and a.only.upper() != k:
            continue
        fn(a.exe)


if __name__ == "__main__":
    main()
