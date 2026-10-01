J07_stratifie_sable_sur_loam : infiltration submergée dans un profil à deux couches (sable 0–30 cm / loam 30–100 cm)
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d ; dt initial 1e-5 d, dt max 0.01 d.
  Water flow parameters : 2 matériaux (Carsel & Parrish) : 1 = sable, 2 = loam.
  Boundary conditions : Upper = Constant pressure head (h = 0) ; Lower = Free drainage.
  Profile : 101 noeuds ; matériau 1 de 0 à -30 cm, matériau 2 en dessous ; h initial = -200 cm.
  Observation nodes : 10, 25, 35, 60 cm.
Résultat attendu : ralentissement du front à l'interface (barrière capillaire ou limitation
par la couche fine) ; comparer les flux vTop et les profils h(z), theta(z) des deux cas.
