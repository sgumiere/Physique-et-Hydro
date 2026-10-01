J05_permanent_nappe_evaporation : régime permanent, loam (Carsel & Parrish), colonne 0–100 cm
Interface HYDRUS-1D :
  Main processes : Water flow. Time units : days ; length : cm. Final time 200 d.
  Water flow parameters : loam (thr 0.078, ths 0.43, alpha 0.036, n 1.56, Ks 24.96, l 0.5).
  Boundary conditions : Upper = Constant flux (0.03 cm/j, positif = évaporation) ;
                        Lower = Constant pressure head (h = 0 : nappe au bas de la colonne).
  Profile : 101 noeuds (dz = 1 cm), condition initiale hydrostatique h(z) = -100 - z (H = -100 cm).
  Observation nodes : 10, 30, 50, 80 cm. Print times : tous les 20 jours.
Résultat attendu : profil h(z) stationnaire à t = 200 j, à comparer avec l'intégration
de dh/dz = -1 - q/K(h) (Buckingham–Darcy en régime permanent).
