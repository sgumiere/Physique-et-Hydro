J07_croute_sans_croute : pluie de 10 cm/j pendant 0.5 j sur un loam sans croûte de battance
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d.
  Water flow parameters : 1 matériau : loam.
  Boundary conditions : Upper = Atmospheric BC with surface run off (hCritS = 0) ; Lower = Free drainage.
  ATMOSPH.IN : t = 0.5 (Prec = 10), t = 1.0 (Prec = 0).
  Profile : 101 noeuds (dz = 1 cm) ; h initial = -300 cm.
Résultat attendu : aucun ruissellement (10 cm/j < Ks), toute la pluie s infiltre.
