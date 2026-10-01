J06_infiltration_submergee_loam : infiltration submergée dans un loam (HYDRUS Simulation 5.1 de Radcliffe & Šimůnek)
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 1 d ; initial time step 1e-4 d ; max 0.02 d.
  Water flow parameters : loam (0.078, 0.43, 0.036, 1.56, 24.96, 0.5).
  Boundary conditions : Upper = Constant pressure head (h = 0 : lame d'eau nulle) ;
                        Lower = Free drainage.
  Profile : 101 noeuds, dz = 1 cm ; h initial = -100 cm partout (h = 0 au noeud de surface).
  Observation nodes : 10, 30, 50 cm. Print times : 0.05, 0.10, ..., 1.0 d.
Résultat attendu : front d'humectation atteignant le bas vers t ≈ 0.6 d ; flux en surface
tendant vers Ks = 24.96 cm/j ; infiltration cumulée ≈ 25.8 cm à 1 d.
