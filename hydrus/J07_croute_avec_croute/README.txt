J07_croute_avec_croute : pluie de 10 cm/j pendant 0.5 j sur un loam avec croûte de battance
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d.
  Water flow parameters : 2 matériaux : 1 = croûte (loam avec Ks = 0.5 cm/j), 2 = loam.
  Boundary conditions : Upper = Atmospheric BC with surface run off (hCritS = 0) ; Lower = Free drainage.
  ATMOSPH.IN : t = 0.5 (Prec = 10), t = 1.0 (Prec = 0).
  Profile : 201 noeuds (dz = 0.5 cm), croûte de 0 à -1 cm ; h initial = -300 cm.
Résultat attendu : ruissellement important dès les premières minutes, infiltration limitée par la croûte.
