"""Source des notebooks du Jour 1 — phase solide et relations masse–volume (style linéaire)."""
from pathlib import Path
import numpy as np
import pandas as pd
from nbbuild import Notebook, ROOT


def make_data():
    d = ROOT / "J01" / "data"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(101)
    rows = []
    for parcelle, rb0, S0 in [("temoin", 1.32, 0.50), ("trafiquee", 1.55, 0.62)]:
        for i, prof in enumerate(["0-10", "10-20", "20-30", "30-40", "40-50", "50-60"]):
            rb = rb0 + 0.03 * i + rng.normal(0, 0.03)
            rs = 2.65 - (0.05 if i < 2 else 0.0)
            n = 1 - rb / rs
            S = np.clip(S0 + 0.045 * i + rng.normal(0, 0.04), 0.4, 0.95)
            theta = S * n
            Ms = rb * 100
            Mt = Ms + theta * 100
            rows.append(dict(id=f"{parcelle[:3].upper()}-{i+1}", parcelle=parcelle, profondeur_cm=prof, V_t_cm3=100.0,
                             M_t_g=round(Mt, 1), M_s_g=round(Ms, 1), rho_s_gcm3=rs,
                             sable_pct=round(38 + rng.normal(0, 3), 0), argile_pct=round(21 + rng.normal(0, 2), 0)))
    df = pd.DataFrame(rows)
    df["limon_pct"] = 100 - df["sable_pct"] - df["argile_pct"]
    df.to_csv(d / "J01_echantillons.csv", index=False)

    dmm = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.063, 0.1, 0.25, 0.5, 1.0, 2.0]
    g = pd.DataFrame({
        "d_mm": dmm,
        "sol_A_pct": [8, 18, 30, 38, 46, 62, 66, 75, 88, 96, 99, 100],       # loam
        "sol_B_pct": [1, 3, 4, 5, 6, 9, 12, 20, 60, 90, 98, 100],            # sable
        "sol_C_pct": [22, 38, 52, 62, 72, 88, 91, 95, 98, 99.5, 100, 100],   # loam limono-argileux
    })
    g.to_csv(d / "J01_granulometrie.csv", index=False)


def build():
    make_data()
    nb = Notebook("J01", "Phase solide et relations masse–volume",
                  "Atelier du Jour 1 : relations masse–volume, granulométrie, loi de Stokes, triangle textural")

    nb.md("""
## Mise en place

Les données sont dans le dossier `data/`. Exécutez la cellule suivante pour importer les bibliothèques.
""")
    nb.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
from scipy.special import erf

rho_w = 1.0   # masse volumique de l'eau, g/cm³
""")

    # ================================================================== Exercice 1
    nb.exercice(
        "Relations masse–volume", duree="15 min",
        enonce="""
Le fichier `data/J01_echantillons.csv` contient 12 échantillons prélevés au cylindre ($V_t = 100$ cm³) dans deux
parcelles (`temoin` et `trafiquee`), à six profondeurs (tranches de 10 cm). Pour chacun : masse humide $M_t$,
masse sèche $M_s$ (105 °C, 24 h), masse volumique des solides $\\rho_s$.

1. Calculer $\\rho_b$, $w$, $\\theta$, $n$, $e$, $S$ et $\\theta_a$ pour chaque échantillon. Rappels :
   $\\rho_b = M_s/V_t$, $w = M_w/M_s$, $\\theta = w\\,\\rho_b/\\rho_w$, $n = 1-\\rho_b/\\rho_s$, $e = n/(1-n)$, $S = \\theta/n$, $\\theta_a = n-\\theta$.
2. Comparer les deux parcelles (moyennes par parcelle, graphique de $\\rho_b$ et $\\theta_a$ en fonction de la profondeur).
   La parcelle trafiquée est-elle compactée ? Quels échantillons ont $\\theta_a < 0{,}10$ ?
3. Calculer le stock d'eau (mm) du profil 0–60 cm de chaque parcelle, chaque échantillon représentant une tranche de 10 cm.
""",
        etapes=[
            dict(titre="Lecture des données", solution="""
df = pd.read_csv("data/J01_echantillons.csv")
print(df)
"""),
            dict(titre="1. Relations masse–volume", solution="""
# masse d'eau = masse humide - masse sèche
df["M_w"] = df["M_t_g"] - df["M_s_g"]

# masse volumique apparente (g/cm³) et teneur en eau massique (-)
df["rho_b"] = df["M_s_g"] / df["V_t_cm3"]
df["w"] = df["M_w"] / df["M_s_g"]

# teneur en eau volumique, porosité, indice des vides, saturation, porosité d'aération
df["theta"] = df["w"] * df["rho_b"] / rho_w
df["n"] = 1 - df["rho_b"] / df["rho_s_gcm3"]
df["e"] = df["n"] / (1 - df["n"])
df["S"] = df["theta"] / df["n"]
df["theta_a"] = df["n"] - df["theta"]

colonnes = ["id", "parcelle", "profondeur_cm", "rho_b", "w", "theta", "n", "e", "S", "theta_a"]
print(df[colonnes].round(3))
""", squelette="""
# masse d'eau = masse humide - masse sèche
df["M_w"] = df["M_t_g"] - df["M_s_g"]

# masse volumique apparente (g/cm³) et teneur en eau massique (-)
df["rho_b"] = # À COMPLÉTER
df["w"] = # À COMPLÉTER

# teneur en eau volumique, porosité, indice des vides, saturation, porosité d'aération
df["theta"] = # À COMPLÉTER
df["n"] = # À COMPLÉTER
df["e"] = # À COMPLÉTER
df["S"] = # À COMPLÉTER
df["theta_a"] = # À COMPLÉTER

colonnes = ["id", "parcelle", "profondeur_cm", "rho_b", "w", "theta", "n", "e", "S", "theta_a"]
print(df[colonnes].round(3))
"""),
            dict(titre="2. Comparaison des deux parcelles", solution="""
# moyennes par parcelle
moyennes = df.groupby("parcelle")[["rho_b", "n", "theta", "theta_a"]].mean()
print(moyennes.round(3))

# profondeur du milieu de chaque tranche (cm) pour le graphique
df["z_cm"] = [5, 15, 25, 35, 45, 55, 5, 15, 25, 35, 45, 55]
temoin = df[df["parcelle"] == "temoin"]
trafiquee = df[df["parcelle"] == "trafiquee"]

plt.figure()
plt.plot(temoin["rho_b"], temoin["z_cm"], "o-", label="témoin")
plt.plot(trafiquee["rho_b"], trafiquee["z_cm"], "s-", label="trafiquée")
plt.gca().invert_yaxis()
plt.xlabel("masse volumique apparente rho_b (g/cm³)")
plt.ylabel("profondeur (cm)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(temoin["theta_a"], temoin["z_cm"], "o-", label="témoin")
plt.plot(trafiquee["theta_a"], trafiquee["z_cm"], "s-", label="trafiquée")
plt.axvline(0.10, color="red", linestyle="--", label="seuil 0,10")
plt.gca().invert_yaxis()
plt.xlabel("porosité d'aération theta_a (-)")
plt.ylabel("profondeur (cm)")
plt.legend()
plt.grid(True)
plt.show()

# échantillons mal aérés
mal_aeres = df[df["theta_a"] < 0.10]
print("Échantillons avec theta_a < 0,10 :", list(mal_aeres["id"]))
""", squelette="""
# moyennes par parcelle
moyennes = # À COMPLÉTER (groupby sur "parcelle", moyenne de rho_b, n, theta, theta_a)
print(moyennes.round(3))

# profondeur du milieu de chaque tranche (cm) pour le graphique
df["z_cm"] = [5, 15, 25, 35, 45, 55, 5, 15, 25, 35, 45, 55]
temoin = df[df["parcelle"] == "temoin"]
trafiquee = df[df["parcelle"] == "trafiquee"]

plt.figure()
plt.plot(temoin["rho_b"], temoin["z_cm"], "o-", label="témoin")
plt.plot(trafiquee["rho_b"], trafiquee["z_cm"], "s-", label="trafiquée")
plt.gca().invert_yaxis()
plt.xlabel("masse volumique apparente rho_b (g/cm³)")
plt.ylabel("profondeur (cm)")
plt.legend()
plt.grid(True)
plt.show()

# À COMPLÉTER : même graphique pour theta_a, avec le seuil 0,10 (plt.axvline)

# échantillons mal aérés
mal_aeres = # À COMPLÉTER
print("Échantillons avec theta_a < 0,10 :", list(mal_aeres["id"]))
"""),
            dict(titre="3. Stock d'eau du profil 0–60 cm", solution="""
# chaque échantillon représente une tranche de 100 mm : stock = somme de theta * 100 mm
stock_temoin = temoin["theta"].sum() * 100
stock_trafiquee = trafiquee["theta"].sum() * 100
print("Stock d'eau 0-60 cm, témoin    :", round(stock_temoin, 1), "mm")
print("Stock d'eau 0-60 cm, trafiquée :", round(stock_trafiquee, 1), "mm")
""", squelette="""
# chaque échantillon représente une tranche de 100 mm : stock = somme de theta * 100 mm
stock_temoin = # À COMPLÉTER
stock_trafiquee = # À COMPLÉTER
print("Stock d'eau 0-60 cm, témoin    :", round(stock_temoin, 1), "mm")
print("Stock d'eau 0-60 cm, trafiquée :", round(stock_trafiquee, 1), "mm")
"""),
        ],
        commentaire="""
La parcelle trafiquée a une $\\rho_b$ supérieure d'environ 0,2 g/cm³ et une porosité inférieure de ~0,08 ; à teneur en eau
comparable, sa porosité d'aération tombe sous le seuil de 0,10 pour plusieurs échantillons : c'est le diagnostic classique de la compaction.
Le stock d'eau est exprimé en mm pour être comparé directement aux pluies et à l'ET (voir Jour 8).
""")

    # ================================================================== Exercice 2
    nb.exercice(
        "Courbes granulométriques", duree="15 min",
        enonce="""
`data/J01_granulometrie.csv` donne le pourcentage passant (masse cumulée) en fonction du diamètre pour trois sols A, B, C.

1. Tracer les trois courbes cumulées (axe des diamètres en échelle logarithmique).
2. Calculer, par **interpolation linéaire en $\\log d$**, les diamètres $D_{10}$, $D_{30}$, $D_{50}$, $D_{60}$ de chaque sol,
   puis $C_u = D_{60}/D_{10}$ et $C_c = D_{30}^2/(D_{10} D_{60})$.
3. Calculer les fractions argile/limon/sable selon l'USDA (2 µm, 50 µm) et selon ISO (2 µm, 63 µm).
4. Ajuster une loi log-normale $F(d) = \\tfrac12[1+\\mathrm{erf}((\\ln d - \\ln d_g)/(\\sqrt2 \\ln\\sigma_g))]$ sur le sol A
   (`curve_fit`) et superposer la courbe ajustée.
""",
        etapes=[
            dict(titre="Lecture des données", solution="""
g = pd.read_csv("data/J01_granulometrie.csv")
d = g["d_mm"].to_numpy()          # diamètres (mm)
pA = g["sol_A_pct"].to_numpy()    # % passant du sol A
pB = g["sol_B_pct"].to_numpy()
pC = g["sol_C_pct"].to_numpy()
print(g)
"""),
            dict(titre="1. Courbes granulométriques", solution="""
plt.figure()
plt.semilogx(d, pA, "o-", label="sol A")
plt.semilogx(d, pB, "s-", label="sol B")
plt.semilogx(d, pC, "^-", label="sol C")
plt.axvline(0.002, color="gray", linestyle=":")    # limite argile / limon (2 µm)
plt.axvline(0.05, color="gray", linestyle=":")     # limite limon / sable USDA (50 µm)
plt.xlabel("diamètre équivalent d (mm)")
plt.ylabel("% passant")
plt.legend()
plt.grid(True)
plt.show()
"""),
            dict(titre="2. Diamètres caractéristiques, Cu et Cc", solution="""
# diamètre tel que x % de la masse est plus fine : interpolation linéaire de log10(d) en fonction de p
def D_x(d, p, x):
    log_d = np.interp(x, p, np.log10(d))
    return 10 ** log_d

for nom, p in [("A", pA), ("B", pB), ("C", pC)]:
    D10 = D_x(d, p, 10)
    D30 = D_x(d, p, 30)
    D50 = D_x(d, p, 50)
    D60 = D_x(d, p, 60)
    Cu = D60 / D10
    Cc = D30**2 / (D10 * D60)
    print(f"sol {nom} : D10 = {D10:.4f} mm, D50 = {D50:.3f} mm, D60 = {D60:.3f} mm, Cu = {Cu:.1f}, Cc = {Cc:.2f}")
""", squelette="""
# diamètre tel que x % de la masse est plus fine : interpolation linéaire de log10(d) en fonction de p
def D_x(d, p, x):
    log_d = # À COMPLÉTER (np.interp)
    return 10 ** log_d

for nom, p in [("A", pA), ("B", pB), ("C", pC)]:
    D10 = D_x(d, p, 10)
    D30 = # À COMPLÉTER
    D50 = # À COMPLÉTER
    D60 = # À COMPLÉTER
    Cu = # À COMPLÉTER
    Cc = # À COMPLÉTER
    print(f"sol {nom} : D10 = {D10:.4f} mm, D50 = {D50:.3f} mm, D60 = {D60:.3f} mm, Cu = {Cu:.1f}, Cc = {Cc:.2f}")
"""),
            dict(titre="3. Fractions texturales USDA et ISO", solution="""
log_d = np.log10(d)
fractions = []
for nom, p in [("A", pA), ("B", pB), ("C", pC)]:
    # % passant à 2 µm, 50 µm et 63 µm (interpolation en log d)
    p_2um = np.interp(np.log10(0.002), log_d, p)
    p_50um = np.interp(np.log10(0.050), log_d, p)
    p_63um = np.interp(np.log10(0.063), log_d, p)
    argile = p_2um
    limon_usda = p_50um - p_2um
    sable_usda = 100 - p_50um
    limon_iso = p_63um - p_2um
    sable_iso = 100 - p_63um
    fractions.append([nom, argile, limon_usda, sable_usda, limon_iso, sable_iso])

frac = pd.DataFrame(fractions, columns=["sol", "argile", "limon_USDA", "sable_USDA", "limon_ISO", "sable_ISO"])
print(frac.round(1))
""", squelette="""
log_d = np.log10(d)
fractions = []
for nom, p in [("A", pA), ("B", pB), ("C", pC)]:
    # % passant à 2 µm, 50 µm et 63 µm (interpolation en log d)
    p_2um = np.interp(np.log10(0.002), log_d, p)
    p_50um = # À COMPLÉTER
    p_63um = # À COMPLÉTER
    argile = # À COMPLÉTER
    limon_usda = # À COMPLÉTER
    sable_usda = # À COMPLÉTER
    limon_iso = # À COMPLÉTER
    sable_iso = # À COMPLÉTER
    fractions.append([nom, argile, limon_usda, sable_usda, limon_iso, sable_iso])

frac = pd.DataFrame(fractions, columns=["sol", "argile", "limon_USDA", "sable_USDA", "limon_ISO", "sable_ISO"])
print(frac.round(1))
"""),
            dict(titre="4. Loi log-normale ajustée sur le sol A", solution="""
# % passant d'une distribution log-normale de diamètre médian dg et d'écart-type géométrique sg
def F_lognormale(d, dg, sg):
    return 50 * (1 + erf((np.log(d) - np.log(dg)) / (np.sqrt(2) * np.log(sg))))

popt, pcov = curve_fit(F_lognormale, d, pA, p0=[0.02, 5])
dg = popt[0]
sg = popt[1]
print(f"dg = {dg * 1000:.1f} µm, sigma_g = {sg:.2f}")

dd = np.logspace(-3.2, 0.4, 200)
plt.figure()
plt.semilogx(d, pA, "o", label="sol A mesuré")
plt.semilogx(dd, F_lognormale(dd, dg, sg), "-", label="log-normale ajustée")
plt.xlabel("d (mm)")
plt.ylabel("% passant")
plt.legend()
plt.grid(True)
plt.show()

erreur = F_lognormale(d, dg, sg) - pA
rmse = np.sqrt(np.mean(erreur**2))
print(f"RMSE de l'ajustement : {rmse:.2f} %")
""", squelette="""
# % passant d'une distribution log-normale de diamètre médian dg et d'écart-type géométrique sg
def F_lognormale(d, dg, sg):
    return # À COMPLÉTER

popt, pcov = curve_fit(F_lognormale, d, pA, p0=[0.02, 5])
dg = popt[0]
sg = popt[1]
print(f"dg = {dg * 1000:.1f} µm, sigma_g = {sg:.2f}")

dd = np.logspace(-3.2, 0.4, 200)
plt.figure()
plt.semilogx(d, pA, "o", label="sol A mesuré")
plt.semilogx(dd, F_lognormale(dd, dg, sg), "-", label="log-normale ajustée")
plt.xlabel("d (mm)")
plt.ylabel("% passant")
plt.legend()
plt.grid(True)
plt.show()

erreur = F_lognormale(d, dg, sg) - pA
rmse = # À COMPLÉTER
print(f"RMSE de l'ajustement : {rmse:.2f} %")
"""),
        ],
        commentaire="""
Le sol B (sable) est uniforme ($C_u \\approx 2$–3), le sol A est bien gradué. Passer de la limite USDA (50 µm) à la limite ISO (63 µm)
déplace quelques pour cent du sable vers le limon : la classe texturale peut changer, d'où l'importance de préciser le système.
L'ajustement log-normal est raisonnable pour A mais une distribution bimodale ferait mieux pour un sol à deux populations de grains.
""")

    # ================================================================== Exercice 3
    nb.exercice(
        "Loi de Stokes et sédimentation", duree="10 min",
        enonce="""
1. Calculer la viscosité de l'eau (Pa·s) par interpolation dans la table : 5 °C : 1,519 ; 10 °C : 1,307 ; 15 °C : 1,139 ;
   20 °C : 1,002 ; 25 °C : 0,890 ; 30 °C : 0,798 (mPa·s).
2. Écrire `stokes(d, T)` qui retourne la vitesse de chute (m/s) d'une particule de diamètre $d$ (m) à la température $T$,
   avec $\\rho_s = 2650$ et $\\rho_w = 998$ kg/m³ : $v = g(\\rho_s-\\rho_w)d^2/(18\\mu)$.
3. Produire la table des temps (minutes) nécessaires pour qu'une particule de 50, 20, 5 et 2 µm parcoure 10 cm, à 15, 20 et 25 °C.
4. Déterminer le diamètre pour lequel $Re = \\rho_w v d/\\mu = 1$ à 20 °C (limite de validité) avec `brentq`.
""",
        etapes=[
            dict(titre="1. Viscosité de l'eau", solution="""
T_table = np.array([5, 10, 15, 20, 25, 30])                               # °C
mu_table = np.array([1.519, 1.307, 1.139, 1.002, 0.890, 0.798]) * 1e-3    # Pa s

# viscosité dynamique (Pa s) par interpolation linéaire dans la table
def viscosite(T):
    return np.interp(T, T_table, mu_table)

print("mu(20 °C) =", viscosite(20), "Pa s")
""", squelette="""
T_table = np.array([5, 10, 15, 20, 25, 30])                               # °C
mu_table = np.array([1.519, 1.307, 1.139, 1.002, 0.890, 0.798]) * 1e-3    # Pa s

# viscosité dynamique (Pa s) par interpolation linéaire dans la table
def viscosite(T):
    return # À COMPLÉTER

print("mu(20 °C) =", viscosite(20), "Pa s")
"""),
            dict(titre="2. Vitesse de chute de Stokes", solution="""
rho_s = 2650.0    # kg/m³
rho_eau = 998.0   # kg/m³
g_pes = 9.81      # m/s²

# vitesse de chute (m/s) d'une sphère de diamètre d (m) à la température T (°C)
def stokes(d, T):
    mu = viscosite(T)
    v = g_pes * (rho_s - rho_eau) * d**2 / (18 * mu)
    return v

print("v(20 µm, 20 °C) =", stokes(20e-6, 20), "m/s")
""", squelette="""
rho_s = 2650.0    # kg/m³
rho_eau = 998.0   # kg/m³
g_pes = 9.81      # m/s²

# vitesse de chute (m/s) d'une sphère de diamètre d (m) à la température T (°C)
def stokes(d, T):
    mu = viscosite(T)
    v = # À COMPLÉTER
    return v

print("v(20 µm, 20 °C) =", stokes(20e-6, 20), "m/s")
"""),
            dict(titre="3. Temps de prélèvement à 10 cm", solution="""
z = 0.10   # profondeur de prélèvement (m)
print("d (µm)   15 °C    20 °C    25 °C   (temps en minutes pour parcourir 10 cm)")
for d_um in [50, 20, 5, 2]:
    d_m = d_um * 1e-6
    t15 = z / stokes(d_m, 15) / 60
    t20 = z / stokes(d_m, 20) / 60
    t25 = z / stokes(d_m, 25) / 60
    print(f"{d_um:5d}   {t15:7.1f}  {t20:7.1f}  {t25:7.1f}")
""", squelette="""
z = 0.10   # profondeur de prélèvement (m)
print("d (µm)   15 °C    20 °C    25 °C   (temps en minutes pour parcourir 10 cm)")
for d_um in [50, 20, 5, 2]:
    d_m = d_um * 1e-6
    t15 = # À COMPLÉTER (temps = distance / vitesse, en minutes)
    t20 = # À COMPLÉTER
    t25 = # À COMPLÉTER
    print(f"{d_um:5d}   {t15:7.1f}  {t20:7.1f}  {t25:7.1f}")
"""),
            dict(titre="4. Limite de validité : Re = 1", solution="""
# nombre de Reynolds de la particule à 20 °C, moins 1 : la racine donne le diamètre limite
def reynolds_moins_1(d):
    v = stokes(d, 20)
    Re = rho_eau * v * d / viscosite(20)
    return Re - 1

d_lim = brentq(reynolds_moins_1, 1e-6, 1e-3)
print(f"Re = 1 pour d = {d_lim * 1e6:.0f} µm : au-delà, la loi de Stokes surestime la vitesse (tamisage nécessaire).")
""", squelette="""
# nombre de Reynolds de la particule à 20 °C, moins 1 : la racine donne le diamètre limite
def reynolds_moins_1(d):
    v = stokes(d, 20)
    Re = # À COMPLÉTER
    return Re - 1

d_lim = brentq(reynolds_moins_1, 1e-6, 1e-3)
print(f"Re = 1 pour d = {d_lim * 1e6:.0f} µm : au-delà, la loi de Stokes surestime la vitesse (tamisage nécessaire).")
"""),
        ],
        commentaire="""
La viscosité varie de ~2,5 %/°C : une erreur de 5 °C sur la température change les temps de prélèvement de ~13 %.
Le diamètre limite (~100 µm) justifie de tamiser les sables et de ne sédimenter que les limons et argiles.
""")

    # ================================================================== Exercice 4
    nb.exercice(
        "Triangle textural", duree="15 min",
        enonce="""
1. Programmer `classe_usda(sable, argile)` qui retourne la classe texturale USDA (12 classes) à partir des règles du triangle
   (le limon vaut $100 - $ sable $-$ argile). Rappels des règles principales : argile ≥ 40 → *argile* (sable ≤ 45, limon < 40),
   *argile limoneuse* (limon ≥ 40) ou *argile sableuse* (argile ≥ 35, sable > 45) ; 27 ≤ argile < 40 → *loam argileux*
   (20 < sable ≤ 45) ou *loam limono-argileux* (sable ≤ 20) ; 20 ≤ argile < 35, sable > 45, limon < 28 → *loam sablo-argileux* ;
   7 ≤ argile < 27, 28 ≤ limon < 50, sable ≤ 52 → *loam* ; limon ≥ 50 → *loam limoneux* (12 ≤ argile < 27, ou argile < 12 et limon < 80)
   ou *limon* (limon ≥ 80, argile < 12) ; sinon : *sable* si limon + 1,5 argile < 15, *sable loameux* si limon + 2 argile < 30,
   *loam sableux* autrement.
2. Convertir une composition en coordonnées cartésiennes du triangle ($x = 100 - $ sable $-$ argile$/2$, $y = $ argile $\\cdot\\sqrt3/2$)
   et tracer le contour du triangle.
3. Placer les sols A, B, C (exercice 2, fractions USDA) et les 12 échantillons de l'exercice 1 ; afficher leur classe.
""",
        etapes=[
            dict(titre="1. Classe texturale USDA", solution="""
def classe_usda(sable, argile):
    limon = 100 - sable - argile
    if argile >= 40 and sable <= 45 and limon < 40:
        return "argile"
    elif argile >= 40 and limon >= 40:
        return "argile limoneuse"
    elif argile >= 35 and sable > 45:
        return "argile sableuse"
    elif 27 <= argile < 40 and 20 < sable <= 45:
        return "loam argileux"
    elif 27 <= argile < 40 and sable <= 20:
        return "loam limono-argileux"
    elif 20 <= argile < 35 and sable > 45 and limon < 28:
        return "loam sablo-argileux"
    elif 7 <= argile < 27 and 28 <= limon < 50 and sable <= 52:
        return "loam"
    elif limon >= 50 and 12 <= argile < 27:
        return "loam limoneux"
    elif 50 <= limon < 80 and argile < 12:
        return "loam limoneux"
    elif limon >= 80 and argile < 12:
        return "limon"
    elif limon + 1.5 * argile < 15:
        return "sable"
    elif limon + 2 * argile < 30:
        return "sable loameux"
    else:
        return "loam sableux"

print(classe_usda(40, 20))   # attendu : loam
print(classe_usda(90, 3))    # attendu : sable
""", squelette="""
def classe_usda(sable, argile):
    limon = 100 - sable - argile
    if argile >= 40 and sable <= 45 and limon < 40:
        return "argile"
    elif argile >= 40 and limon >= 40:
        return "argile limoneuse"
    # À COMPLÉTER : les autres règles, dans l'ordre de l'énoncé (elif ... : return "...")
    else:
        return "loam sableux"

print(classe_usda(40, 20))   # attendu : loam
print(classe_usda(90, 3))    # attendu : sable
"""),
            dict(titre="2. Coordonnées et contour du triangle", solution="""
# coordonnées cartésiennes d'un point du triangle (sable en bas, argile en haut)
def x_tri(sable, argile):
    return 100 - sable - argile / 2

def y_tri(sable, argile):
    return argile * np.sqrt(3) / 2

# les trois sommets : 100 % sable, 100 % limon, 100 % argile
x_contour = []
y_contour = []
for s, a in [(100, 0), (0, 0), (0, 100), (100, 0)]:
    x_contour.append(x_tri(s, a))
    y_contour.append(y_tri(s, a))

plt.figure(figsize=(7, 6))
plt.plot(x_contour, y_contour, "k-")
plt.text(0, -5, "100 % sable", ha="center")
plt.text(100, -5, "100 % limon", ha="center")
plt.text(50, 90, "100 % argile", ha="center")
plt.axis("equal")
plt.axis("off")
plt.show()
""", squelette="""
# coordonnées cartésiennes d'un point du triangle (sable en bas, argile en haut)
def x_tri(sable, argile):
    return # À COMPLÉTER

def y_tri(sable, argile):
    return # À COMPLÉTER

x_contour = []
y_contour = []
for s, a in [(100, 0), (0, 0), (0, 100), (100, 0)]:
    x_contour.append(x_tri(s, a))
    y_contour.append(y_tri(s, a))

plt.figure(figsize=(7, 6))
plt.plot(x_contour, y_contour, "k-")
plt.text(0, -5, "100 % sable", ha="center")
plt.text(100, -5, "100 % limon", ha="center")
plt.text(50, 90, "100 % argile", ha="center")
plt.axis("equal")
plt.axis("off")
plt.show()
"""),
            dict(titre="3. Placer les sols et les échantillons", solution="""
plt.figure(figsize=(7, 6))
plt.plot(x_contour, y_contour, "k-")

# sols A, B, C de l'exercice 2 (fractions USDA)
for i in range(len(frac)):
    nom = frac.loc[i, "sol"]
    s = frac.loc[i, "sable_USDA"]
    a = frac.loc[i, "argile"]
    classe = classe_usda(s, a)
    plt.plot(x_tri(s, a), y_tri(s, a), "o", markersize=10)
    plt.text(x_tri(s, a) + 2, y_tri(s, a) + 2, "sol " + nom + " : " + classe)
    print("sol", nom, ":", classe)

# les 12 échantillons de l'exercice 1
for i in range(len(df)):
    s = df.loc[i, "sable_pct"]
    a = df.loc[i, "argile_pct"]
    plt.plot(x_tri(s, a), y_tri(s, a), "g^")

plt.text(0, -5, "100 % sable", ha="center")
plt.text(100, -5, "100 % limon", ha="center")
plt.text(50, 90, "100 % argile", ha="center")
plt.axis("equal")
plt.axis("off")
plt.show()

# classe de chaque échantillon
classes = []
for i in range(len(df)):
    classes.append(classe_usda(df.loc[i, "sable_pct"], df.loc[i, "argile_pct"]))
df["classe_USDA"] = classes
print(df["classe_USDA"].value_counts())
""", squelette="""
plt.figure(figsize=(7, 6))
plt.plot(x_contour, y_contour, "k-")

# sols A, B, C de l'exercice 2 (fractions USDA)
for i in range(len(frac)):
    nom = frac.loc[i, "sol"]
    s = frac.loc[i, "sable_USDA"]
    a = frac.loc[i, "argile"]
    classe = # À COMPLÉTER
    plt.plot(x_tri(s, a), y_tri(s, a), "o", markersize=10)
    plt.text(x_tri(s, a) + 2, y_tri(s, a) + 2, "sol " + nom + " : " + classe)
    print("sol", nom, ":", classe)

# À COMPLÉTER : placer les 12 échantillons de l'exercice 1 (colonnes sable_pct et argile_pct de df) avec plt.plot(..., "g^")

plt.text(0, -5, "100 % sable", ha="center")
plt.text(100, -5, "100 % limon", ha="center")
plt.text(50, 90, "100 % argile", ha="center")
plt.axis("equal")
plt.axis("off")
plt.show()

# classe de chaque échantillon
classes = []
for i in range(len(df)):
    classes.append(# À COMPLÉTER)
df["classe_USDA"] = classes
print(df["classe_USDA"].value_counts())
"""),
        ],
        commentaire="""
Les 12 échantillons (38 % sable, 21 % argile) sont tous des loams ; le sol C (argile ~38 %) est un loam limono-argileux et le sol B un sable.
""")

    # ================================================================== Bonus
    nb.exercice(
        "Bonus — vérification numérique de l'analyse dimensionnelle", duree="facultatif",
        enonce="""
Le théorème de Buckingham appliqué à la chute d'une particule donne deux nombres sans dimension,
$\\Pi_1 = v\\,d\\,\\Delta\\rho/\\mu$ et $\\Pi_2 = g\\,d^3\\,\\Delta\\rho^2/\\mu^2$. Vérifier numériquement, pour $d$ de 1 à 60 µm,
que $\\Pi_1/\\Pi_2 = 1/18$ dans le régime de Stokes (à 20 °C).
""",
        etapes=[
            dict(titre="Rapport Π1/Π2", solution="""
mu = viscosite(20)
delta_rho = rho_s - rho_eau
for d_um in [1, 10, 20, 40, 60]:
    d_m = d_um * 1e-6
    v = stokes(d_m, 20)
    Pi1 = v * d_m * delta_rho / mu
    Pi2 = g_pes * d_m**3 * delta_rho**2 / mu**2
    print(f"d = {d_um:2d} µm : Pi1/Pi2 = {Pi1 / Pi2:.5f}")
print("1/18 =", round(1 / 18, 5))
""", squelette="""
mu = viscosite(20)
delta_rho = rho_s - rho_eau
for d_um in [1, 10, 20, 40, 60]:
    d_m = d_um * 1e-6
    v = stokes(d_m, 20)
    Pi1 = # À COMPLÉTER
    Pi2 = # À COMPLÉTER
    print(f"d = {d_um:2d} µm : Pi1/Pi2 = {Pi1 / Pi2:.5f}")
print("1/18 =", round(1 / 18, 5))
"""),
        ],
        commentaire="Le rapport Π₁/Π₂ = 1/18 est exact par construction de la loi de Stokes : les deux nombres sans dimension ne sont pas indépendants dans ce régime.")

    nb.md("""
## Pour aller plus loin

* Refaire l'exercice 2 avec une distribution bimodale (somme de deux log-normales) pour le sol C.
* Estimer la surface spécifique des trois sols à partir de leur courbe granulométrique ($S_s = \\sum_i f_i\\, 6/(\\rho_s d_i)$).
""")
    return nb
