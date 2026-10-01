J06_pluie_flux_impose_loam : pluie de 30 cm/j (> Ks = 24.96 cm/j) pendant 0.5 j, puis arrêt
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d.
  Boundary conditions : Upper = Atmospheric BC with surface run off (hCritS = 0) ;
                        Lower = Free drainage.
  Time-variable BC (ATMOSPH.IN) : 2 enregistrements : t = 0.5 (Prec = 30), t = 1.0 (Prec = 0).
  Profile : 101 noeuds, h initial = -300 cm. Observation nodes 5, 20, 50 cm.
Résultat attendu : vTop = rTop tant que la capacité d'infiltration > 30 cm/j ; ensuite h_top = 0,
vTop < rTop et RunOff > 0 (T_LEVEL.OUT). Comparer le temps de submersion à Green–Ampt.
