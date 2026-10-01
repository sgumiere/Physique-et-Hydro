J08_redistribution_loam : drainage interne (« méthode du profil couvert ») d'un loam de 100 cm, 30 j
Interface HYDRUS-1D :
  Main processes : Water flow. Final time 30 d ; print times chaque jour.
  Boundary conditions : Upper = Constant flux = 0 (surface couverte : ni pluie ni évaporation) ;
                        Lower = Free drainage.
  Profile : 101 noeuds (0 à -100 cm) ; h initial = -10 cm partout (theta ≈ 0.41, proche de la saturation).
  Observation nodes : 10, 25, 50, 90 cm.
Résultat attendu : flux de drainage vBot décroissant rapidement (loi de puissance), teneur en eau
tendant vers la « capacité au champ » dynamique ; comparer theta(t) à 25 cm avec theta(h = -100 cm) et
theta(h = -330 cm) ; stock (Volume) et drainage cumulé sum(vBot) dans T_LEVEL.OUT.
