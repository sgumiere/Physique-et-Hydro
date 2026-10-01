J09_chaleur_sans_convection : onde de température journalière (20 ± 10 °C) sans convection dans un loam
Interface HYDRUS-1D :
  Main processes : Water flow + Heat transport. Final time 5 d ; print times toutes les 3 h.
  Water flow : loam ; Upper = Constant flux 0 ; Lower = Constant flux 0 ; h initial = -100 cm (theta ≈ 0.243).
  Heat transport parameters : Chung & Horton, loam (b1 1.56728e19, b2 2.53474e19, b3 9.89388e19 ;
      Cn 1.43327e17, C0 1.8737e17, Cw 3.12035e17 ; fraction solide 0.57 ; dispersivité thermique 5 cm).
  Heat BC : Upper = Temperature BC (Dirichlet) : moyenne 20 °C, amplitude 10 °C, période 1 j ;
            Lower = Zero gradient. Température initiale 20 °C.
  Observation nodes : 5, 10, 20, 40 cm.
Résultat attendu : amortissement exponentiel de l'amplitude et déphasage croissant avec la profondeur ;
comparer à la solution analytique T(z,t) = Tm + A exp(-z/d) sin(omega t - z/d), d = sqrt(2 D_T/omega).
