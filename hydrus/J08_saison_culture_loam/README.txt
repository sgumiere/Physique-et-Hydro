J08_saison_culture_loam : saison de culture de maïs (120 j) sur un loam, avec prélèvement racinaire (Feddes)
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
