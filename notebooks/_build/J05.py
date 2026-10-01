"""Source des notebooks du Jour 5 — écoulement non saturé en régime permanent et introduction à HYDRUS-1D."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from nbbuild import Notebook, ROOT

# Paramètres de van Genuchten–Mualem (Carsel & Parrish 1988), cm et jours
SOLS = {
    "sable":         [0.045, 0.430, 0.145, 2.68, 712.8, 0.5],
    "loam sableux":  [0.065, 0.410, 0.075, 1.89, 106.1, 0.5],
    "loam":          [0.078, 0.430, 0.036, 1.56, 24.96, 0.5],
    "loam limoneux": [0.067, 0.450, 0.020, 1.41, 10.80, 0.5],
    "argile":        [0.068, 0.380, 0.008, 1.09, 4.80, 0.5],
}


def _vg_K(h, p):
    thr, ths, a, n, Ks, l = p
    m = 1 - 1 / n
    Se = (1 + (a * abs(h)) ** n) ** (-m) if h < 0 else 1.0
    return Ks * Se ** l * (1 - (1 - Se ** (1 / m)) ** m) ** 2


def make_data():
    d = ROOT / "J05" / "data"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(505)
    rows = []
    r = 10.0
    for site, texture, fK in [("A", "sable", 0.7), ("B", "loam sableux", 1.4), ("C", "loam", 1.1), ("D", "loam limoneux", 0.8)]:
        p = SOLS[texture]
        lc = quad(lambda h: _vg_K(h, p) / p[4], -1e4, 0, limit=400, points=[-1000, -100, -10, -1])[0]
        aG = 1 / lc
        # Ks « de terrain » : le catalogue à un facteur près (0,7–1,4), alpha_G aussi (0,85–1,15)
        Ks = p[4] * fK
        aGf = aG * rng.uniform(0.85, 1.15)
        row = dict(site=site, texture=texture, r_cm=r)
        for h0 in (-3.0, -10.0):
            Q = np.pi * r**2 * Ks * np.exp(aGf * h0) * (1 + 4 / (np.pi * r * aGf))   # cm3/j
            Q *= 1 + rng.normal(0, 0.03)
            row[f"Q_h{int(-h0)}_cm3min"] = round(Q / 1440, 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(d / "J05_infiltrometre.csv", index=False)


def build():
    make_data()
    nb = Notebook("J05", "Écoulement non saturé permanent et introduction à HYDRUS-1D",
                  "Atelier du Jour 5 : fonctions hydrauliques, profils permanents, lecture d'un projet HYDRUS-1D, infiltromètre à disque")

    nb.md("""
## Mise en place

Les projets HYDRUS-1D de référence sont dans `../../hydrus/` et se lisent avec le module `hydrus_io.py` (dossier `notebooks/`).
Unités du cours : **cm** et **jours** ; flux **positifs vers le haut** (convention HYDRUS-1D) ; profondeurs négatives.
Exécutez la cellule suivante pour importer les bibliothèques et définir la table des sols de Carsel & Parrish (1988).
""")
    nb.code("""
import sys
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from hydrus_io import read_tlevel, read_nod_inf, read_obs_node, read_balance, read_run_inf

HYD = "../../hydrus"     # dossier des projets HYDRUS-1D

# sols de Carsel & Parrish (1988) : thr, ths (-), alpha (1/cm), n (-), Ks (cm/j) ; l = 0,5 pour tous
sols = pd.DataFrame(index=["sable", "loam sableux", "loam", "loam limoneux", "argile"])
sols["thr"] = [0.045, 0.065, 0.078, 0.067, 0.068]
sols["ths"] = [0.430, 0.410, 0.430, 0.450, 0.380]
sols["alpha"] = [0.145, 0.075, 0.036, 0.020, 0.008]
sols["n"] = [2.68, 1.89, 1.56, 1.41, 1.09]
sols["Ks"] = [712.8, 106.1, 24.96, 10.80, 4.80]
print(sols)
""")

    # ================================================================== Exercice 1
    nb.exercice(
        "Fonctions hydrauliques de Mualem–van Genuchten", duree="15 min",
        enonce="""
1. Écrire les fonctions `vg_theta(h, thr, ths, alpha, n)`, `vg_K(h, thr, ths, alpha, n, Ks)` et `vg_C(h, thr, ths, alpha, n)`
   (valables pour $h \\le 0$, avec $l = 0{,}5$) :
   $S_e = [1+(\\alpha|h|)^n]^{-m}$, $m = 1-1/n$ ; $\\theta = \\theta_r + (\\theta_s-\\theta_r)S_e$ ;
   $K = K_s S_e^{l}[1-(1-S_e^{1/m})^m]^2$ ; $C = (\\theta_s-\\theta_r)\\,\\alpha n m (\\alpha|h|)^{n-1}[1+(\\alpha|h|)^n]^{-m-1}$.
   Vérifier le loam : $K(-50) \\approx 0{,}258$ cm/j, $K(-500) \\approx 1{,}7\\times10^{-4}$ cm/j.
2. Tracer $K(h)$ et $C(h)$ en fonction de $|h|$ (échelles log-log) pour le sable, le loam et l'argile, puis $D(\\theta) = K/C$ (échelle log en $y$).
3. Calculer la longueur capillaire macroscopique $\\lambda_c = K_s^{-1}\\int_{-\\infty}^{0} K(h)\\,dh$ pour les cinq sols
   (trapèzes sur une grille logarithmique de $|h|$, `np.trapezoid`) et $\\alpha_G = 1/\\lambda_c$. Vérifier le loam : $\\lambda_c \\approx 6{,}9$ cm.
""",
        etapes=[
            dict(titre="1. Fonctions de van Genuchten et Mualem", solution="""
# teneur en eau de van Genuchten (1980), h <= 0 : Se = [1 + (alpha |h|)^n]^(-m), m = 1 - 1/n
def vg_theta(h, thr, ths, alpha, n):
    m = 1 - 1 / n
    Se = (1 + (alpha * np.abs(h)) ** n) ** (-m)
    theta = thr + (ths - thr) * Se
    return theta

# conductivité hydraulique de Mualem–van Genuchten (cm/j), l = 0,5
def vg_K(h, thr, ths, alpha, n, Ks):
    m = 1 - 1 / n
    Se = (1 + (alpha * np.abs(h)) ** n) ** (-m)
    K = Ks * Se ** 0.5 * (1 - (1 - Se ** (1 / m)) ** m) ** 2
    return K

# capacité capillaire C = d theta / dh (1/cm)
def vg_C(h, thr, ths, alpha, n):
    m = 1 - 1 / n
    ah = alpha * np.abs(h)
    C = (ths - thr) * alpha * n * m * ah ** (n - 1) * (1 + ah ** n) ** (-m - 1)
    return C

# vérification sur le loam
thr = sols.loc["loam", "thr"]
ths = sols.loc["loam", "ths"]
alpha = sols.loc["loam", "alpha"]
n = sols.loc["loam", "n"]
Ks = sols.loc["loam", "Ks"]
print(f"loam : K(-50) = {vg_K(-50, thr, ths, alpha, n, Ks):.4f} cm/j")
print(f"loam : K(-500) = {vg_K(-500, thr, ths, alpha, n, Ks):.3e} cm/j")
print(f"loam : theta(-50) = {vg_theta(-50, thr, ths, alpha, n):.3f}")
""", squelette="""
# teneur en eau de van Genuchten (1980), h <= 0 : Se = [1 + (alpha |h|)^n]^(-m), m = 1 - 1/n
def vg_theta(h, thr, ths, alpha, n):
    m = 1 - 1 / n
    Se = (1 + (alpha * np.abs(h)) ** n) ** (-m)
    theta = # À COMPLÉTER
    return theta

# conductivité hydraulique de Mualem–van Genuchten (cm/j), l = 0,5
def vg_K(h, thr, ths, alpha, n, Ks):
    m = 1 - 1 / n
    Se = (1 + (alpha * np.abs(h)) ** n) ** (-m)
    K = # À COMPLÉTER
    return K

# capacité capillaire C = d theta / dh (1/cm)
def vg_C(h, thr, ths, alpha, n):
    m = 1 - 1 / n
    ah = alpha * np.abs(h)
    C = # À COMPLÉTER
    return C

# vérification sur le loam
thr = sols.loc["loam", "thr"]
ths = sols.loc["loam", "ths"]
alpha = sols.loc["loam", "alpha"]
n = sols.loc["loam", "n"]
Ks = sols.loc["loam", "Ks"]
print(f"loam : K(-50) = {vg_K(-50, thr, ths, alpha, n, Ks):.4f} cm/j")
print(f"loam : K(-500) = {vg_K(-500, thr, ths, alpha, n, Ks):.3e} cm/j")
print(f"loam : theta(-50) = {vg_theta(-50, thr, ths, alpha, n):.3f}")
"""),
            dict(titre="2. Courbes K(h), C(h) et D(θ)", solution="""
h_abs = np.logspace(-1, 4, 300)     # |h| de 0,1 à 10 000 cm, grille logarithmique
h_grille = -h_abs

plt.figure()
for nom in ["sable", "loam", "argile"]:
    s = sols.loc[nom]
    K = vg_K(h_grille, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"])
    plt.loglog(h_abs, K, label=nom)
plt.ylim(1e-8, 1e3)
plt.xlabel("|h| (cm)")
plt.ylabel("K (cm/j)")
plt.title("conductivité hydraulique K(h)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for nom in ["sable", "loam", "argile"]:
    s = sols.loc[nom]
    C = vg_C(h_grille, s["thr"], s["ths"], s["alpha"], s["n"])
    plt.loglog(h_abs, C, label=nom)
plt.xlabel("|h| (cm)")
plt.ylabel("C (1/cm)")
plt.title("capacité capillaire C(h)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for nom in ["sable", "loam", "argile"]:
    s = sols.loc[nom]
    theta = vg_theta(h_grille, s["thr"], s["ths"], s["alpha"], s["n"])
    K = vg_K(h_grille, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"])
    C = vg_C(h_grille, s["thr"], s["ths"], s["alpha"], s["n"])
    D = K / C
    plt.semilogy(theta, D, label=nom)
plt.ylim(1e-1, 1e6)
plt.xlabel("theta (-)")
plt.ylabel("D = K/C (cm²/j)")
plt.title("diffusivité D(theta)")
plt.legend()
plt.grid(True)
plt.show()
""", squelette="""
h_abs = np.logspace(-1, 4, 300)     # |h| de 0,1 à 10 000 cm, grille logarithmique
h_grille = -h_abs

plt.figure()
for nom in ["sable", "loam", "argile"]:
    s = sols.loc[nom]
    K = vg_K(h_grille, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"])
    plt.loglog(h_abs, K, label=nom)
plt.ylim(1e-8, 1e3)
plt.xlabel("|h| (cm)")
plt.ylabel("K (cm/j)")
plt.title("conductivité hydraulique K(h)")
plt.legend()
plt.grid(True)
plt.show()

# À COMPLÉTER : même graphique pour C(h) (plt.loglog)

# À COMPLÉTER : diffusivité D = K / C en fonction de theta (plt.semilogy, plt.ylim(1e-1, 1e6))
"""),
            dict(titre="3. Longueur capillaire macroscopique", solution="""
h_abs = np.logspace(-2, 4, 2000)     # grille logarithmique de |h| : 0,01 à 10 000 cm
lambda_c = []
alpha_G = []
theta_100 = []
K_100 = []
for nom in sols.index:
    s = sols.loc[nom]
    K = vg_K(-h_abs, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"])
    # intégrale de K/Ks par trapèzes, plus le segment [0 ; 0,01 cm] où K = Ks
    lc = np.trapezoid(K / s["Ks"], h_abs) + 0.01
    lambda_c.append(lc)
    alpha_G.append(1 / lc)
    theta_100.append(vg_theta(-100, s["thr"], s["ths"], s["alpha"], s["n"]))
    K_100.append(vg_K(-100, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"]))
sols["lambda_c"] = lambda_c
sols["alpha_G"] = alpha_G
sols["theta_100"] = theta_100
sols["K_100"] = K_100
print(sols.round(4))
""", squelette="""
h_abs = np.logspace(-2, 4, 2000)     # grille logarithmique de |h| : 0,01 à 10 000 cm
lambda_c = []
alpha_G = []
theta_100 = []
K_100 = []
for nom in sols.index:
    s = sols.loc[nom]
    K = vg_K(-h_abs, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"])
    # intégrale de K/Ks par trapèzes, plus le segment [0 ; 0,01 cm] où K = Ks
    lc = # À COMPLÉTER (np.trapezoid)
    lambda_c.append(lc)
    alpha_G.append(# À COMPLÉTER)
    theta_100.append(vg_theta(-100, s["thr"], s["ths"], s["alpha"], s["n"]))
    K_100.append(vg_K(-100, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"]))
sols["lambda_c"] = lambda_c
sols["alpha_G"] = alpha_G
sols["theta_100"] = theta_100
sols["K_100"] = K_100
print(sols.round(4))
"""),
        ],
        commentaire="""
$K$ chute de 8 à 10 ordres de grandeur entre la saturation et $|h| = 10^4$ cm ; les courbes se croisent vers $|h| \\approx 30$–100 cm :
au-delà, le sable est moins conducteur que l'argile. $\\lambda_c$ croît du sable (3,8 cm) au loam limoneux (9 cm) : la capillarité
gagne en importance dans les sols fins. La valeur MvG de l'argile ($n = 1{,}09$) est peu réaliste (queue de $K$ très longue, $K_r$
non intégrable en pratique) : c'est une limite connue du modèle pour $n < 1{,}2$.
""")

    # ================================================================== Exercice 2
    nb.exercice(
        "Profils permanents au-dessus d'une nappe et évaporation maximale", duree="20 min",
        enonce="""
En régime permanent le flux $q$ est constant et $\\dfrac{dh}{dz} = -1 - \\dfrac{q}{K(h)}$ ($z$ vers le haut, $q > 0$ vers le haut).
On intègre cette équation **depuis la nappe** ($h = 0$ en $z = -L$) vers la surface par petits pas $dz$ (schéma d'Euler) :
$h_{i+1} = h_i + \\left(-1 - \\dfrac{q}{K(h_i)}\\right) dz$.

1. Loam, nappe à $L = 100$ cm, infiltration $q = -1$ cm/j : calculer le profil $h(z)$ avec $dz = 0{,}1$ cm et $h$ en surface
   (attendu $\\approx -28{,}6$ cm).
2. Même calcul pour l'évaporation $q = +0{,}03$ cm/j (attendu $h \\approx -138$ cm en surface).
3. Lire `NOD_INF.OUT` des projets `J05_permanent_nappe_infiltration` et `J05_permanent_nappe_evaporation` (`read_nod_inf`),
   extraire le profil à $t = 200$ j, le superposer à votre calcul et calculer l'écart RMS sur $h$ (HYDRUS : $h$ en surface
   $= -28{,}9$ cm et $-135{,}2$ cm).
4. Évaporation maximale : la profondeur de nappe maximale compatible avec un flux $q$ est
   $L_{max}(q) = \\int_{-\\infty}^{0} \\dfrac{dh}{1 + q/K(h)} = \\int_{-\\infty}^{0} \\dfrac{K}{K + q}\\,dh$ (Gardner 1958).
   Calculer $L_{max}$ par trapèzes, puis $e_{max}$ (le $q$ tel que $L_{max}(q) = L$, `brentq`) pour le loam et $L = 100$ cm
   (attendu $\\approx 0{,}054$ cm/j).
5. Tracer $e_{max}(L)$ pour $L$ de 30 à 300 cm pour le sable, le loam et le loam limoneux (échelles log) et estimer l'exposant
   de la loi de puissance $e_{max} \\propto L^{-n}$ entre 70 et 200 cm.
""",
        etapes=[
            dict(titre="1. Profil d'infiltration permanente (q = −1 cm/j)", solution="""
# paramètres du loam
thr = sols.loc["loam", "thr"]
ths = sols.loc["loam", "ths"]
alpha = sols.loc["loam", "alpha"]
n = sols.loc["loam", "n"]
Ks = sols.loc["loam", "Ks"]

L = 100.0      # profondeur de la nappe (cm)
dz = 0.1       # pas d'intégration (cm)
n_pas = round(L / dz)
q = -1.0       # flux (cm/j), négatif = vers le bas (infiltration)

# départ à la nappe (z = -L, h = 0) ; on monte vers la surface pas à pas
z = -L
h = 0.0
z_inf = [z]
h_inf = [h]
for i in range(n_pas):
    K = vg_K(h, thr, ths, alpha, n, Ks)
    dh_dz = -1 - q / K
    h = h + dh_dz * dz
    z = z + dz
    z_inf.append(z)
    h_inf.append(h)
z_inf = np.array(z_inf)
h_inf = np.array(h_inf)

print(f"infiltration q = {q} cm/j : h en surface = {h_inf[-1]:.2f} cm")
print(f"h à 50 cm au-dessus de la nappe = {np.interp(-50, z_inf, h_inf):.2f} cm (gradient unitaire : K(h) = |q|)")
""", squelette="""
# paramètres du loam
thr = sols.loc["loam", "thr"]
ths = sols.loc["loam", "ths"]
alpha = sols.loc["loam", "alpha"]
n = sols.loc["loam", "n"]
Ks = sols.loc["loam", "Ks"]

L = 100.0      # profondeur de la nappe (cm)
dz = 0.1       # pas d'intégration (cm)
n_pas = round(L / dz)
q = -1.0       # flux (cm/j), négatif = vers le bas (infiltration)

# départ à la nappe (z = -L, h = 0) ; on monte vers la surface pas à pas
z = -L
h = 0.0
z_inf = [z]
h_inf = [h]
for i in range(n_pas):
    K = # À COMPLÉTER
    dh_dz = # À COMPLÉTER
    h = # À COMPLÉTER (schéma d'Euler)
    z = z + dz
    z_inf.append(z)
    h_inf.append(h)
z_inf = np.array(z_inf)
h_inf = np.array(h_inf)

print(f"infiltration q = {q} cm/j : h en surface = {h_inf[-1]:.2f} cm")
print(f"h à 50 cm au-dessus de la nappe = {np.interp(-50, z_inf, h_inf):.2f} cm (gradient unitaire : K(h) = |q|)")
"""),
            dict(titre="2. Profil d'évaporation permanente (q = +0,03 cm/j)", solution="""
q = 0.03       # flux (cm/j), positif = vers le haut (évaporation)

z = -L
h = 0.0
z_eva = [z]
h_eva = [h]
for i in range(n_pas):
    K = vg_K(h, thr, ths, alpha, n, Ks)
    dh_dz = -1 - q / K
    h = h + dh_dz * dz
    z = z + dz
    z_eva.append(z)
    h_eva.append(h)
z_eva = np.array(z_eva)
h_eva = np.array(h_eva)

print(f"évaporation q = {q} cm/j : h en surface = {h_eva[-1]:.2f} cm")
""", squelette="""
q = 0.03       # flux (cm/j), positif = vers le haut (évaporation)

z = -L
h = 0.0
z_eva = [z]
h_eva = [h]
for i in range(n_pas):
    K = # À COMPLÉTER
    dh_dz = # À COMPLÉTER
    h = # À COMPLÉTER
    z = z + dz
    z_eva.append(z)
    h_eva.append(h)
z_eva = np.array(z_eva)
h_eva = np.array(h_eva)

print(f"évaporation q = {q} cm/j : h en surface = {h_eva[-1]:.2f} cm")
"""),
            dict(titre="3. Comparaison avec les profils HYDRUS-1D à t = 200 j", solution="""
nod_inf = read_nod_inf(HYD + "/J05_permanent_nappe_infiltration")
nod_eva = read_nod_inf(HYD + "/J05_permanent_nappe_evaporation")
print("temps d'impression (j) :", list(nod_inf.keys()))
prof_inf = nod_inf[200.0]        # profil au dernier temps d'impression (régime permanent)
prof_eva = nod_eva[200.0]
print(prof_inf[["Depth", "Head", "Moisture", "K", "Flux"]].head())

# h en surface d'après HYDRUS (noeud 1, Depth = 0)
h_hyd_inf = prof_inf["Head"].iloc[0]
h_hyd_eva = prof_eva["Head"].iloc[0]

# profil calculé interpolé aux profondeurs des noeuds HYDRUS, puis écart RMS
h_calc_inf = np.interp(prof_inf["Depth"], z_inf, h_inf)
rms_inf = np.sqrt(np.mean((h_calc_inf - prof_inf["Head"]) ** 2))
h_calc_eva = np.interp(prof_eva["Depth"], z_eva, h_eva)
rms_eva = np.sqrt(np.mean((h_calc_eva - prof_eva["Head"]) ** 2))
print(f"infiltration : h surface Euler = {h_inf[-1]:.2f} cm, HYDRUS = {h_hyd_inf:.2f} cm, écart RMS = {rms_inf:.3f} cm")
print(f"évaporation : h surface Euler = {h_eva[-1]:.2f} cm, HYDRUS = {h_hyd_eva:.2f} cm, écart RMS = {rms_eva:.3f} cm")

plt.figure()
plt.plot(h_inf, z_inf, label="Euler, q = -1 cm/j")
plt.plot(prof_inf["Head"], prof_inf["Depth"], "o", markersize=3, label="HYDRUS-1D, t = 200 j")
plt.plot([0, -100], [-100, 0], "k:", label="hydrostatique (t = 0)")
plt.xlabel("h (cm)")
plt.ylabel("z (cm)")
plt.title("loam, nappe à 100 cm : infiltration")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(h_eva, z_eva, label="Euler, q = +0,03 cm/j")
plt.plot(prof_eva["Head"], prof_eva["Depth"], "o", markersize=3, label="HYDRUS-1D, t = 200 j")
plt.plot([0, -100], [-100, 0], "k:", label="hydrostatique (t = 0)")
plt.xlabel("h (cm)")
plt.ylabel("z (cm)")
plt.title("loam, nappe à 100 cm : évaporation")
plt.legend()
plt.grid(True)
plt.show()
""", squelette="""
nod_inf = read_nod_inf(HYD + "/J05_permanent_nappe_infiltration")
nod_eva = read_nod_inf(HYD + "/J05_permanent_nappe_evaporation")
print("temps d'impression (j) :", list(nod_inf.keys()))
prof_inf = nod_inf[200.0]        # profil au dernier temps d'impression (régime permanent)
prof_eva = nod_eva[200.0]
print(prof_inf[["Depth", "Head", "Moisture", "K", "Flux"]].head())

# h en surface d'après HYDRUS (noeud 1, Depth = 0)
h_hyd_inf = prof_inf["Head"].iloc[0]
h_hyd_eva = prof_eva["Head"].iloc[0]

# profil calculé interpolé aux profondeurs des noeuds HYDRUS, puis écart RMS
h_calc_inf = np.interp(prof_inf["Depth"], z_inf, h_inf)
rms_inf = # À COMPLÉTER
h_calc_eva = # À COMPLÉTER
rms_eva = # À COMPLÉTER
print(f"infiltration : h surface Euler = {h_inf[-1]:.2f} cm, HYDRUS = {h_hyd_inf:.2f} cm, écart RMS = {rms_inf:.3f} cm")
print(f"évaporation : h surface Euler = {h_eva[-1]:.2f} cm, HYDRUS = {h_hyd_eva:.2f} cm, écart RMS = {rms_eva:.3f} cm")

plt.figure()
plt.plot(h_inf, z_inf, label="Euler, q = -1 cm/j")
plt.plot(prof_inf["Head"], prof_inf["Depth"], "o", markersize=3, label="HYDRUS-1D, t = 200 j")
plt.plot([0, -100], [-100, 0], "k:", label="hydrostatique (t = 0)")
plt.xlabel("h (cm)")
plt.ylabel("z (cm)")
plt.title("loam, nappe à 100 cm : infiltration")
plt.legend()
plt.grid(True)
plt.show()

# À COMPLÉTER : même graphique pour l'évaporation
"""),
            dict(titre="4. Évaporation maximale depuis une nappe à 100 cm (Gardner 1958)", solution="""
# profondeur maximale de nappe (cm) capable d'entretenir le flux q (cm/j) : L_max = intégrale de K/(K + q) dh
def L_max(q, thr, ths, alpha, n, Ks):
    h_abs = np.logspace(-2, 7, 4000)
    K = vg_K(-h_abs, thr, ths, alpha, n, Ks)
    L = np.trapezoid(K / (K + q), h_abs) + 0.01
    return L

# écart entre L_max(q) et la profondeur L de la nappe : il s'annule pour q = e_max
def ecart_L(q, L, thr, ths, alpha, n, Ks):
    return L_max(q, thr, ths, alpha, n, Ks) - L

for q in [0.01, 0.03, 0.05, 0.10]:
    print(f"loam, q = {q} cm/j : L_max = {L_max(q, thr, ths, alpha, n, Ks):.1f} cm")

# brentq cherche q entre 1e-12 et Ks tel que ecart_L = 0 ; args = les autres arguments de ecart_L
e_max_loam = brentq(ecart_L, 1e-12, Ks, args=(100.0, thr, ths, alpha, n, Ks))
print(f"e_max du loam, nappe à 100 cm : {e_max_loam:.4f} cm/j")
""", squelette="""
# profondeur maximale de nappe (cm) capable d'entretenir le flux q (cm/j) : L_max = intégrale de K/(K + q) dh
def L_max(q, thr, ths, alpha, n, Ks):
    h_abs = np.logspace(-2, 7, 4000)
    K = vg_K(-h_abs, thr, ths, alpha, n, Ks)
    L = # À COMPLÉTER (np.trapezoid, plus le segment [0 ; 0,01 cm])
    return L

# écart entre L_max(q) et la profondeur L de la nappe : il s'annule pour q = e_max
def ecart_L(q, L, thr, ths, alpha, n, Ks):
    return # À COMPLÉTER

for q in [0.01, 0.03, 0.05, 0.10]:
    print(f"loam, q = {q} cm/j : L_max = {L_max(q, thr, ths, alpha, n, Ks):.1f} cm")

# brentq cherche q entre 1e-12 et Ks tel que ecart_L = 0 ; args = les autres arguments de ecart_L
e_max_loam = # À COMPLÉTER (brentq(ecart_L, 1e-12, Ks, args=(...)))
print(f"e_max du loam, nappe à 100 cm : {e_max_loam:.4f} cm/j")
"""),
            dict(titre="5. Évaporation maximale en fonction de la profondeur de nappe", solution="""
Ls = np.array([30, 50, 70, 100, 150, 200, 300])     # profondeurs de nappe (cm)
tab = pd.DataFrame(index=Ls)
tab.index.name = "L (cm)"

plt.figure()
for nom in ["sable", "loam", "loam limoneux"]:
    s = sols.loc[nom]
    e_max = []
    for L in Ls:
        e = brentq(ecart_L, 1e-12, s["Ks"], args=(L, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"]))
        e_max.append(e)
    tab[nom] = e_max
    plt.loglog(Ls, e_max, "o-", label=nom)
plt.axhline(0.3, color="gray", linestyle=":", label="demande E_p typique (3 mm/j)")
plt.xlabel("profondeur de la nappe L (cm)")
plt.ylabel("e_max (cm/j)")
plt.legend()
plt.grid(True)
plt.show()
print(tab)

# exposant de la loi de puissance e_max ~ L^(-n) entre 70 et 200 cm (pente dans le plan log-log)
for nom in ["sable", "loam", "loam limoneux"]:
    pente = -(np.log(tab.loc[200, nom]) - np.log(tab.loc[70, nom])) / (np.log(200) - np.log(70))
    print(f"{nom} : e_max ~ L^-{pente:.1f} entre 70 et 200 cm")
""", squelette="""
Ls = np.array([30, 50, 70, 100, 150, 200, 300])     # profondeurs de nappe (cm)
tab = pd.DataFrame(index=Ls)
tab.index.name = "L (cm)"

plt.figure()
for nom in ["sable", "loam", "loam limoneux"]:
    s = sols.loc[nom]
    e_max = []
    for L in Ls:
        e = # À COMPLÉTER (brentq avec les paramètres du sol s)
        e_max.append(e)
    tab[nom] = e_max
    plt.loglog(Ls, e_max, "o-", label=nom)
plt.axhline(0.3, color="gray", linestyle=":", label="demande E_p typique (3 mm/j)")
plt.xlabel("profondeur de la nappe L (cm)")
plt.ylabel("e_max (cm/j)")
plt.legend()
plt.grid(True)
plt.show()
print(tab)

# exposant de la loi de puissance e_max ~ L^(-n) entre 70 et 200 cm (pente dans le plan log-log)
for nom in ["sable", "loam", "loam limoneux"]:
    pente = # À COMPLÉTER
    print(f"{nom} : e_max ~ L^-{pente:.1f} entre 70 et 200 cm")
"""),
        ],
        commentaire="""
Pour l'infiltration, le profil devient vertical (gradient unitaire) dès 40–50 cm au-dessus de la nappe, à $h \\approx -28{,}6$ cm
tel que $K(h) = 1$ cm/j ; HYDRUS donne $-28{,}9$ cm (moyenne arithmétique de $K$ entre nœuds près de la nappe). Pour l'évaporation à
0,03 cm/j la surface est à $-135$ cm. L'évaporation maximale du loam pour une nappe à 1 m ($\\approx 0{,}054$ cm/j) est bien
inférieure à une demande de 3 mm/j : l'évaporation est limitée par le sol. La décroissance de $e_{max}$ suit une loi de puissance
$L^{-n}$ avec $n \\approx 3$ pour le loam, comme prévu par Gardner (1958).
""")

    # ================================================================== Exercice 3
    nb.exercice(
        "Lecture d'un projet HYDRUS-1D : convergence vers le régime permanent", duree="15 min",
        enonce="""
Les deux projets `J05_permanent_nappe_*` partent d'un profil hydrostatique ($h = -100 - z$, $H = -100$ cm) et imposent un flux constant
en surface (−1 cm/j ou +0,03 cm/j) avec une nappe fixe ($h = 0$) au bas d'une colonne de loam de 100 cm, pendant 200 j.

1. Lire `T_LEVEL.OUT` des deux projets (`read_tlevel`) et tracer `vTop` et `vBot` (flux en surface et au bas, > 0 vers le haut)
   en fonction du temps.
2. Déterminer $t_{99}$, premier instant où $|$`vBot` − `vTop`$| < 1$ % de $|q|$.
3. Tracer `hTop` (charge de pression en surface) en fonction du temps.
4. Lire `OBS_NODE.OUT` (`read_obs_node`) et tracer $h(t)$ aux nœuds d'observation (10, 30, 50 et 80 cm).
5. Lire `RUN_INF.OUT` (`read_run_inf`) : tracer $\\Delta t$ en fonction du temps (échelle log) et l'histogramme du nombre
   d'itérations par pas ; compter les pas et les itérations totales.
6. Lire `BALANCE.OUT` (`read_balance`) : erreur relative maximale `WatBalR` (%) et variation du stock `W-volume` entre 0 et 200 j ;
   vérifier qu'elle est cohérente avec `sum(vBot)` − `sum(vTop)` de `T_LEVEL.OUT`
   ($\\Delta W = \\int (q_{bas} - q_{haut})\\,dt$ avec $q > 0$ vers le haut).
""",
        etapes=[
            dict(titre="Lecture des fichiers T_LEVEL.OUT", solution="""
tl_inf = read_tlevel(HYD + "/J05_permanent_nappe_infiltration")
tl_eva = read_tlevel(HYD + "/J05_permanent_nappe_evaporation")
print("colonnes :", list(tl_inf.columns))
print(tl_eva[["rTop", "vTop", "vBot", "hTop", "hBot", "sum(vTop)", "sum(vBot)"]].head())
"""),
            dict(titre="1. Flux en surface et au bas de la colonne", solution="""
plt.figure()
plt.plot(tl_inf.index, tl_inf["vTop"], label="vTop (surface)")
plt.plot(tl_inf.index, tl_inf["vBot"], "--", label="vBot (nappe)")
plt.xlim(0, 60)
plt.xlabel("temps (j)")
plt.ylabel("flux (cm/j, > 0 vers le haut)")
plt.title("J05_permanent_nappe_infiltration")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(tl_eva.index, tl_eva["vTop"], label="vTop (surface)")
plt.plot(tl_eva.index, tl_eva["vBot"], "--", label="vBot (nappe)")
plt.xlabel("temps (j)")
plt.ylabel("flux (cm/j, > 0 vers le haut)")
plt.title("J05_permanent_nappe_evaporation")
plt.legend()
plt.grid(True)
plt.show()
""", squelette="""
plt.figure()
plt.plot(tl_inf.index, tl_inf["vTop"], label="vTop (surface)")
plt.plot(tl_inf.index, tl_inf["vBot"], "--", label="vBot (nappe)")
plt.xlim(0, 60)
plt.xlabel("temps (j)")
plt.ylabel("flux (cm/j, > 0 vers le haut)")
plt.title("J05_permanent_nappe_infiltration")
plt.legend()
plt.grid(True)
plt.show()

# À COMPLÉTER : même graphique pour l'évaporation (tl_eva)
"""),
            dict(titre="2. Temps d'atteinte du régime permanent t99", solution="""
# infiltration : flux imposé q = rTop ; écart entre les flux au bas et en surface
q_inf = tl_inf["rTop"].iloc[-1]
temps = tl_inf.index.to_numpy()
ecart = np.abs(tl_inf["vBot"].to_numpy() - tl_inf["vTop"].to_numpy())
seuil = 0.01 * abs(q_inf)
instants = temps[ecart < seuil]       # tous les instants où l'écart est sous 1 % de |q|
t99_inf = instants[0]
print(f"infiltration : q = {q_inf} cm/j, t99 = {t99_inf} j")

# évaporation
q_eva = tl_eva["rTop"].iloc[-1]
temps = tl_eva.index.to_numpy()
ecart = np.abs(tl_eva["vBot"].to_numpy() - tl_eva["vTop"].to_numpy())
seuil = 0.01 * abs(q_eva)
instants = temps[ecart < seuil]
t99_eva = instants[0]
print(f"évaporation : q = {q_eva} cm/j, t99 = {t99_eva} j")
""", squelette="""
# infiltration : flux imposé q = rTop ; écart entre les flux au bas et en surface
q_inf = tl_inf["rTop"].iloc[-1]
temps = tl_inf.index.to_numpy()
ecart = np.abs(tl_inf["vBot"].to_numpy() - tl_inf["vTop"].to_numpy())
seuil = # À COMPLÉTER (1 % de |q|)
instants = temps[ecart < seuil]       # tous les instants où l'écart est sous 1 % de |q|
t99_inf = instants[0]
print(f"infiltration : q = {q_inf} cm/j, t99 = {t99_inf} j")

# évaporation
q_eva = tl_eva["rTop"].iloc[-1]
temps = tl_eva.index.to_numpy()
ecart = # À COMPLÉTER
seuil = # À COMPLÉTER
instants = temps[ecart < seuil]
t99_eva = instants[0]
print(f"évaporation : q = {q_eva} cm/j, t99 = {t99_eva} j")
"""),
            dict(titre="3. Charge de pression en surface", solution="""
plt.figure()
plt.plot(tl_inf.index, tl_inf["hTop"], label="infiltration (q = -1 cm/j)")
plt.plot(tl_eva.index, tl_eva["hTop"], label="évaporation (q = +0,03 cm/j)")
plt.xlabel("temps (j)")
plt.ylabel("hTop (cm)")
plt.title("charge de pression en surface")
plt.legend()
plt.grid(True)
plt.show()

hTop_inf = tl_inf["hTop"].iloc[-1]
hTop_eva = tl_eva["hTop"].iloc[-1]
print(f"hTop à 200 j : infiltration {hTop_inf:.2f} cm, évaporation {hTop_eva:.2f} cm")
"""),
            dict(titre="4. Charges de pression aux nœuds d'observation (OBS_NODE.OUT)", solution="""
obs_inf = read_obs_node(HYD + "/J05_permanent_nappe_infiltration")
obs_eva = read_obs_node(HYD + "/J05_permanent_nappe_evaporation")
print(obs_eva.head(3))

# noeuds 11, 31, 51, 81 = profondeurs 10, 30, 50, 80 cm (dz = 1 cm, noeud 1 en surface)
plt.figure()
for noeud in [11, 31, 51, 81]:
    plt.plot(obs_inf.index, obs_inf[(noeud, "h")], label=f"noeud {noeud} ({noeud - 1} cm)")
plt.xlim(0, 60)
plt.xlabel("temps (j)")
plt.ylabel("h (cm)")
plt.title("infiltration : h aux noeuds d'observation")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for noeud in [11, 31, 51, 81]:
    plt.plot(obs_eva.index, obs_eva[(noeud, "h")], label=f"noeud {noeud} ({noeud - 1} cm)")
plt.xlabel("temps (j)")
plt.ylabel("h (cm)")
plt.title("évaporation : h aux noeuds d'observation")
plt.legend()
plt.grid(True)
plt.show()
""", squelette="""
obs_inf = read_obs_node(HYD + "/J05_permanent_nappe_infiltration")
obs_eva = read_obs_node(HYD + "/J05_permanent_nappe_evaporation")
print(obs_eva.head(3))

# noeuds 11, 31, 51, 81 = profondeurs 10, 30, 50, 80 cm (dz = 1 cm, noeud 1 en surface)
plt.figure()
for noeud in [11, 31, 51, 81]:
    plt.plot(obs_inf.index, obs_inf[(noeud, "h")], label=f"noeud {noeud} ({noeud - 1} cm)")
plt.xlim(0, 60)
plt.xlabel("temps (j)")
plt.ylabel("h (cm)")
plt.title("infiltration : h aux noeuds d'observation")
plt.legend()
plt.grid(True)
plt.show()

# À COMPLÉTER : même graphique pour l'évaporation (obs_eva)
"""),
            dict(titre="5. Pas de temps et itérations (RUN_INF.OUT)", solution="""
ri_inf = read_run_inf(HYD + "/J05_permanent_nappe_infiltration")
ri_eva = read_run_inf(HYD + "/J05_permanent_nappe_evaporation")
print(ri_inf.head(3))

plt.figure()
plt.semilogy(ri_inf["Time"], ri_inf["dt"], label="infiltration")
plt.semilogy(ri_eva["Time"], ri_eva["dt"], label="évaporation")
plt.xlabel("temps (j)")
plt.ylabel("pas de temps dt (j)")
plt.title("RUN_INF.OUT : pas de temps")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.hist(ri_inf["Iter"], bins=np.arange(0.5, 10.5, 1), alpha=0.6, label="infiltration")     # alpha = transparence
plt.hist(ri_eva["Iter"], bins=np.arange(0.5, 10.5, 1), alpha=0.6, label="évaporation")
plt.xlabel("itérations de Picard par pas")
plt.ylabel("nombre de pas")
plt.title("RUN_INF.OUT : itérations")
plt.legend()
plt.grid(True)
plt.show()

# nombre de pas, itérations totales et moyennes, pas de temps min et max
n_pas_inf = len(ri_inf)
iter_total_inf = ri_inf["Iter"].sum()
iter_moy_inf = ri_inf["Iter"].mean()
print(f"infiltration : {n_pas_inf} pas, {iter_total_inf:.0f} itérations ({iter_moy_inf:.2f} par pas), dt de {ri_inf['dt'].min()} à {ri_inf['dt'].max()} j")
n_pas_eva = len(ri_eva)
iter_total_eva = ri_eva["Iter"].sum()
iter_moy_eva = ri_eva["Iter"].mean()
print(f"évaporation : {n_pas_eva} pas, {iter_total_eva:.0f} itérations ({iter_moy_eva:.2f} par pas), dt de {ri_eva['dt'].min()} à {ri_eva['dt'].max()} j")
""", squelette="""
ri_inf = read_run_inf(HYD + "/J05_permanent_nappe_infiltration")
ri_eva = read_run_inf(HYD + "/J05_permanent_nappe_evaporation")
print(ri_inf.head(3))

plt.figure()
plt.semilogy(ri_inf["Time"], ri_inf["dt"], label="infiltration")
plt.semilogy(ri_eva["Time"], ri_eva["dt"], label="évaporation")
plt.xlabel("temps (j)")
plt.ylabel("pas de temps dt (j)")
plt.title("RUN_INF.OUT : pas de temps")
plt.legend()
plt.grid(True)
plt.show()

# À COMPLÉTER : histogramme du nombre d'itérations par pas (plt.hist, bins=np.arange(0.5, 10.5, 1)) pour les deux projets

# nombre de pas, itérations totales et moyennes, pas de temps min et max
n_pas_inf = # À COMPLÉTER
iter_total_inf = # À COMPLÉTER
iter_moy_inf = # À COMPLÉTER
print(f"infiltration : {n_pas_inf} pas, {iter_total_inf:.0f} itérations ({iter_moy_inf:.2f} par pas), dt de {ri_inf['dt'].min()} à {ri_inf['dt'].max()} j")
n_pas_eva = # À COMPLÉTER
iter_total_eva = # À COMPLÉTER
iter_moy_eva = # À COMPLÉTER
print(f"évaporation : {n_pas_eva} pas, {iter_total_eva:.0f} itérations ({iter_moy_eva:.2f} par pas), dt de {ri_eva['dt'].min()} à {ri_eva['dt'].max()} j")
"""),
            dict(titre="6. Bilan de masse (BALANCE.OUT)", solution="""
bal_inf = read_balance(HYD + "/J05_permanent_nappe_infiltration")
bal_eva = read_balance(HYD + "/J05_permanent_nappe_evaporation")
print(bal_eva[["W-volume", "Top Flux", "Bot Flux", "WatBalT", "WatBalR"]])

# erreur relative maximale de bilan (%)
WatBalR_inf = bal_inf["WatBalR"].abs().max()
WatBalR_eva = bal_eva["WatBalR"].abs().max()
print(f"WatBalR max : infiltration {WatBalR_inf:.3f} %, évaporation {WatBalR_eva:.3f} %")

# variation du stock d'eau entre 0 et 200 j (cm) d'après BALANCE.OUT
W0_inf = bal_inf["W-volume"].iloc[0]
W200_inf = bal_inf["W-volume"].iloc[-1]
dW_bilan_inf = W200_inf - W0_inf
W0_eva = bal_eva["W-volume"].iloc[0]
W200_eva = bal_eva["W-volume"].iloc[-1]
dW_bilan_eva = W200_eva - W0_eva

# d'après les flux cumulés de T_LEVEL.OUT : dW = sum(vBot) - sum(vTop)  (q > 0 vers le haut)
dW_flux_inf = tl_inf["sum(vBot)"].iloc[-1] - tl_inf["sum(vTop)"].iloc[-1]
dW_flux_eva = tl_eva["sum(vBot)"].iloc[-1] - tl_eva["sum(vTop)"].iloc[-1]

print(f"infiltration : W0 = {W0_inf:.3f} cm, W200 = {W200_inf:.3f} cm, dW = {dW_bilan_inf:.3f} cm (BALANCE) et {dW_flux_inf:.3f} cm (flux cumulés)")
print(f"évaporation : W0 = {W0_eva:.3f} cm, W200 = {W200_eva:.3f} cm, dW = {dW_bilan_eva:.3f} cm (BALANCE) et {dW_flux_eva:.3f} cm (flux cumulés)")
""", squelette="""
bal_inf = read_balance(HYD + "/J05_permanent_nappe_infiltration")
bal_eva = read_balance(HYD + "/J05_permanent_nappe_evaporation")
print(bal_eva[["W-volume", "Top Flux", "Bot Flux", "WatBalT", "WatBalR"]])

# erreur relative maximale de bilan (%)
WatBalR_inf = # À COMPLÉTER
WatBalR_eva = # À COMPLÉTER
print(f"WatBalR max : infiltration {WatBalR_inf:.3f} %, évaporation {WatBalR_eva:.3f} %")

# variation du stock d'eau entre 0 et 200 j (cm) d'après BALANCE.OUT
W0_inf = bal_inf["W-volume"].iloc[0]
W200_inf = bal_inf["W-volume"].iloc[-1]
dW_bilan_inf = # À COMPLÉTER
W0_eva = bal_eva["W-volume"].iloc[0]
W200_eva = bal_eva["W-volume"].iloc[-1]
dW_bilan_eva = # À COMPLÉTER

# d'après les flux cumulés de T_LEVEL.OUT : dW = sum(vBot) - sum(vTop)  (q > 0 vers le haut)
dW_flux_inf = # À COMPLÉTER
dW_flux_eva = # À COMPLÉTER

print(f"infiltration : W0 = {W0_inf:.3f} cm, W200 = {W200_inf:.3f} cm, dW = {dW_bilan_inf:.3f} cm (BALANCE) et {dW_flux_inf:.3f} cm (flux cumulés)")
print(f"évaporation : W0 = {W0_eva:.3f} cm, W200 = {W200_eva:.3f} cm, dW = {dW_bilan_eva:.3f} cm (BALANCE) et {dW_flux_eva:.3f} cm (flux cumulés)")
"""),
        ],
        commentaire="""
Le régime permanent (à 1 % près) est atteint en ~11 j pour l'infiltration (l'eau infiltrée traverse un sol déjà humide) mais en
~75 j pour l'évaporation, car la diffusivité du sol qui sèche est très faible. Le pas de temps croît de 0,01 j à 1 j (borné par
l'intervalle d'impression de 20 j et par dtMax), avec 2 à 6 itérations de Picard par pas. L'erreur de bilan `WatBalR` reste
inférieure à 0,01 % et la variation de stock (+5,1 cm en infiltration, −0,6 cm en évaporation) est retrouvée à partir des flux
cumulés de `T_LEVEL.OUT`.
""")

    # ================================================================== Exercice 4
    nb.exercice(
        "Infiltromètre à disque : Wooding et méthode à deux tensions", duree="10 min",
        enonce="""
`data/J05_infiltrometre.csv` : pour quatre sites (textures indiquées), débits permanents $Q$ (cm³/min) mesurés avec un disque de rayon
$r = 10$ cm aux tensions $h_0 = -3$ cm et $-10$ cm.

Solution de Wooding (1968) avec $K = K_s e^{\\alpha_G h}$ : $Q = \\pi r^2 K(h_0)\\left(1 + \\dfrac{4}{\\pi r \\alpha_G}\\right)$.

1. Pour chaque site, calculer $\\alpha_G = \\ln(Q_1/Q_2)/(h_1 - h_2)$, $\\lambda_c = 1/\\alpha_G$, puis $K_s$ (convertir $Q$ en cm³/j).
2. Calculer la part du débit due à la capillarité latérale, $\\dfrac{4/(\\pi r\\alpha_G)}{1 + 4/(\\pi r\\alpha_G)}$, à $h_0 = -3$ cm.
3. Comparer $K_s$ et $\\lambda_c$ aux valeurs de Carsel & Parrish de la texture (exercice 1) ; commenter les écarts.
""",
        etapes=[
            dict(titre="Lecture des données", solution="""
inf = pd.read_csv("data/J05_infiltrometre.csv")
print(inf)
"""),
            dict(titre="1. Paramètres de Gardner par la méthode à deux tensions", solution="""
h1 = -3.0       # première tension (cm)
h2 = -10.0      # seconde tension (cm)
Q1 = inf["Q_h3_cm3min"] * 1440      # débit à h1, converti en cm³/j
Q2 = inf["Q_h10_cm3min"] * 1440     # débit à h2, cm³/j
r = inf["r_cm"]

# alpha_G (1/cm) et longueur capillaire (cm) : le rapport des débits ne dépend que de alpha_G
inf["alpha_G"] = np.log(Q1 / Q2) / (h1 - h2)
inf["lambda_c"] = 1 / inf["alpha_G"]

# facteur géométrique de Wooding, puis Ks en extrapolant K(h1) à h = 0
geom = 1 + 4 / (np.pi * r * inf["alpha_G"])
inf["Ks"] = Q1 * np.exp(-inf["alpha_G"] * h1) / (np.pi * r ** 2 * geom)
print(inf[["site", "texture", "alpha_G", "lambda_c", "Ks"]].round(3))
""", squelette="""
h1 = -3.0       # première tension (cm)
h2 = -10.0      # seconde tension (cm)
Q1 = inf["Q_h3_cm3min"] * 1440      # débit à h1, converti en cm³/j
Q2 = inf["Q_h10_cm3min"] * 1440     # débit à h2, cm³/j
r = inf["r_cm"]

# alpha_G (1/cm) et longueur capillaire (cm) : le rapport des débits ne dépend que de alpha_G
inf["alpha_G"] = # À COMPLÉTER
inf["lambda_c"] = # À COMPLÉTER

# facteur géométrique de Wooding, puis Ks en extrapolant K(h1) à h = 0
geom = # À COMPLÉTER
inf["Ks"] = # À COMPLÉTER
print(inf[["site", "texture", "alpha_G", "lambda_c", "Ks"]].round(3))
"""),
            dict(titre="2. Part du débit due à la capillarité latérale", solution="""
# part (%) du débit à h0 = -3 cm qui est due à la capillarité latérale : (geom - 1) / geom
inf["part_capillaire_%"] = 100 * (geom - 1) / geom
print(inf[["site", "texture", "part_capillaire_%"]].round(1))
""", squelette="""
# part (%) du débit à h0 = -3 cm qui est due à la capillarité latérale : (geom - 1) / geom
inf["part_capillaire_%"] = # À COMPLÉTER
print(inf[["site", "texture", "part_capillaire_%"]].round(1))
"""),
            dict(titre="3. Comparaison avec les valeurs de Carsel & Parrish", solution="""
Ks_CP = []
lambda_c_MvG = []
K10_Gardner = []
K10_MvG = []
for i in range(len(inf)):
    texture = inf.loc[i, "texture"]
    s = sols.loc[texture]
    Ks_CP.append(s["Ks"])
    lambda_c_MvG.append(s["lambda_c"])
    # K(-10 cm) selon Gardner (terrain) et selon Mualem–van Genuchten (catalogue)
    K10_Gardner.append(inf.loc[i, "Ks"] * np.exp(inf.loc[i, "alpha_G"] * (-10)))
    K10_MvG.append(vg_K(-10, s["thr"], s["ths"], s["alpha"], s["n"], s["Ks"]))
inf["Ks_CarselParrish"] = Ks_CP
inf["lambda_c_MvG"] = lambda_c_MvG
inf["K10_Gardner"] = K10_Gardner
inf["K10_MvG"] = K10_MvG
colonnes = ["site", "texture", "Ks", "Ks_CarselParrish", "lambda_c", "lambda_c_MvG", "K10_Gardner", "K10_MvG"]
print(inf[colonnes].round(3))

# courbes K(h) de Gardner estimées sur chaque site, avec les deux points de mesure
h_grille = np.linspace(-15, 0, 50)
plt.figure()
for i in range(len(inf)):
    Ks_site = inf.loc[i, "Ks"]
    aG_site = inf.loc[i, "alpha_G"]
    etiquette = "site " + inf.loc[i, "site"] + " (" + inf.loc[i, "texture"] + ")"
    plt.semilogy(h_grille, Ks_site * np.exp(aG_site * h_grille), label=etiquette)
    plt.plot([h1, h2], [Ks_site * np.exp(aG_site * h1), Ks_site * np.exp(aG_site * h2)], "ko", markersize=4)
plt.xlabel("h (cm)")
plt.ylabel("K(h) (cm/j)")
plt.title("K(h) de Gardner estimée par la méthode à deux tensions")
plt.legend()
plt.grid(True)
plt.show()
""", squelette="""
Ks_CP = []
lambda_c_MvG = []
K10_Gardner = []
K10_MvG = []
for i in range(len(inf)):
    texture = inf.loc[i, "texture"]
    s = sols.loc[texture]
    Ks_CP.append(s["Ks"])
    lambda_c_MvG.append(s["lambda_c"])
    # K(-10 cm) selon Gardner (terrain) et selon Mualem–van Genuchten (catalogue)
    K10_Gardner.append(# À COMPLÉTER)
    K10_MvG.append(# À COMPLÉTER)
inf["Ks_CarselParrish"] = Ks_CP
inf["lambda_c_MvG"] = lambda_c_MvG
inf["K10_Gardner"] = K10_Gardner
inf["K10_MvG"] = K10_MvG
colonnes = ["site", "texture", "Ks", "Ks_CarselParrish", "lambda_c", "lambda_c_MvG", "K10_Gardner", "K10_MvG"]
print(inf[colonnes].round(3))

# courbes K(h) de Gardner estimées sur chaque site, avec les deux points de mesure
h_grille = np.linspace(-15, 0, 50)
plt.figure()
for i in range(len(inf)):
    Ks_site = inf.loc[i, "Ks"]
    aG_site = inf.loc[i, "alpha_G"]
    etiquette = "site " + inf.loc[i, "site"] + " (" + inf.loc[i, "texture"] + ")"
    plt.semilogy(h_grille, Ks_site * np.exp(aG_site * h_grille), label=etiquette)
    plt.plot([h1, h2], [Ks_site * np.exp(aG_site * h1), Ks_site * np.exp(aG_site * h2)], "ko", markersize=4)
plt.xlabel("h (cm)")
plt.ylabel("K(h) (cm/j)")
plt.title("K(h) de Gardner estimée par la méthode à deux tensions")
plt.legend()
plt.grid(True)
plt.show()
"""),
        ],
        commentaire="""
Le facteur géométrique $1 + 4/(\\pi r \\alpha_G)$ vaut 1,4 (sable) à 2,1 (loam limoneux) : négliger la capillarité latérale
surestimerait $K$ de 40 à 110 %. Les $K_s$ « de terrain » diffèrent du catalogue d'un facteur 0,7 à 1,4 (variabilité typique,
souvent bien plus au champ) alors que $\\lambda_c$ est retrouvée à ±20 % : la longueur capillaire est une propriété plus stable que $K_s$.
Rappel : ces mesures excluent les macropores ($h_0 < 0$) ; $K_s$ estimée par extrapolation à $h = 0$ est celle de la matrice.
""")

    # ================================================================== Bonus
    nb.exercice(
        "Bonus — solution analytique de Gardner pour l'évaporation maximale", duree="facultatif",
        enonce="""
Pour $K = K_s e^{\\alpha_G h}$, on montre que $L_{max}(q) = \\dfrac{1}{\\alpha_G}\\ln\\!\\left(1 + \\dfrac{K_s}{q}\\right)$ et donc
$e_{max}(L) = \\dfrac{K_s}{e^{\\alpha_G L} - 1}$. Calculer $e_{max}$ du loam ($K_s = 24{,}96$ cm/j, $\\alpha_G = 1/\\lambda_c$) pour
$L = 30, 50, 100$ cm et comparer aux valeurs MvG de l'exercice 2. D'où vient l'écart ?
""",
        etapes=[
            dict(titre="e_max de Gardner et de Mualem–van Genuchten", solution="""
# paramètres du loam ; alpha_G = 1 / lambda_c (exercice 1)
thr = sols.loc["loam", "thr"]
ths = sols.loc["loam", "ths"]
alpha = sols.loc["loam", "alpha"]
n = sols.loc["loam", "n"]
Ks = sols.loc["loam", "Ks"]
aG = sols.loc["loam", "alpha_G"]

# L_max = intégrale de dh / (1 + (q/Ks) e^(-aG h)) ; avec u = e^(aG h) : L_max = ln(1 + Ks/q) / aG, donc e_max = Ks / (e^(aG L) - 1)
for L in [30, 50, 100]:
    e_gardner = Ks / (np.exp(aG * L) - 1)
    e_mvg = brentq(ecart_L, 1e-12, Ks, args=(L, thr, ths, alpha, n, Ks))
    print(f"L = {L:3d} cm : e_max Gardner = {e_gardner:.3e} cm/j, e_max MvG = {e_mvg:.3e} cm/j")

# comparaison des deux K(h) sur la gamme utile
h_abs = np.logspace(0, 3, 200)
plt.figure()
plt.loglog(h_abs, vg_K(-h_abs, thr, ths, alpha, n, Ks), label="Mualem–van Genuchten")
plt.loglog(h_abs, Ks * np.exp(-aG * h_abs), "--", label=f"Gardner, alpha_G = {aG:.3f} 1/cm")
plt.ylim(1e-8, 1e2)
plt.xlabel("|h| (cm)")
plt.ylabel("K (cm/j)")
plt.legend()
plt.grid(True)
plt.show()
""", squelette="""
# paramètres du loam ; alpha_G = 1 / lambda_c (exercice 1)
thr = sols.loc["loam", "thr"]
ths = sols.loc["loam", "ths"]
alpha = sols.loc["loam", "alpha"]
n = sols.loc["loam", "n"]
Ks = sols.loc["loam", "Ks"]
aG = sols.loc["loam", "alpha_G"]

# L_max = intégrale de dh / (1 + (q/Ks) e^(-aG h)) ; avec u = e^(aG h) : L_max = ln(1 + Ks/q) / aG, donc e_max = Ks / (e^(aG L) - 1)
for L in [30, 50, 100]:
    e_gardner = # À COMPLÉTER
    e_mvg = brentq(ecart_L, 1e-12, Ks, args=(L, thr, ths, alpha, n, Ks))
    print(f"L = {L:3d} cm : e_max Gardner = {e_gardner:.3e} cm/j, e_max MvG = {e_mvg:.3e} cm/j")

# comparaison des deux K(h) sur la gamme utile
h_abs = np.logspace(0, 3, 200)
plt.figure()
plt.loglog(h_abs, vg_K(-h_abs, thr, ths, alpha, n, Ks), label="Mualem–van Genuchten")
plt.loglog(h_abs, Ks * np.exp(-aG * h_abs), "--", label=f"Gardner, alpha_G = {aG:.3f} 1/cm")
plt.ylim(1e-8, 1e2)
plt.xlabel("|h| (cm)")
plt.ylabel("K (cm/j)")
plt.legend()
plt.grid(True)
plt.show()
"""),
        ],
        commentaire="""
Le modèle exponentiel sous-estime $e_{max}$ de plusieurs ordres de grandeur dès que $L > 50$ cm : sa conductivité décroît
beaucoup plus vite que la loi de puissance de MvG ($K \\propto |h|^{-3{,}4}$ pour le loam loin de la saturation), or c'est la
queue de $K(h)$ dans le sol sec qui contrôle la remontée capillaire. Gardner (1958) utilisait d'ailleurs $K = a/(b + |h|^n)$
pour ce problème, pas l'exponentielle.
""")

    nb.md("""
## Pour aller plus loin

* Reprendre l'exercice 2 avec la nappe à 50 cm et à 150 cm : comment évolue $h$ en surface pour $q = -1$ cm/j ?
* Dans HYDRUS-1D, remplacer le flux imposé de $+0{,}03$ cm/j par $+0{,}06$ cm/j ($> e_{max}$) et observer `RUN_INF.OUT`.
* Réaliser l'essai à trois tensions ($-3$, $-6$, $-10$ cm) et vérifier la linéarité de $\\ln Q$ en fonction de $h_0$.
""")
    return nb
