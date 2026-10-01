J10_traceur_disp10cm : traceur inerte en régime permanent non saturé (loam sableux, q = 10 cm/j, θ = 0.330)
Interface HYDRUS-1D :
  Main processes : Water flow + Solute transport (equilibrium). Final time 6 d ; print times 0.5 j.
  Water flow : loam sableux (0.065, 0.41, 0.075, 1.89, 106.1, 0.5) ; Upper = Constant flux -10 cm/j ;
               Lower = Free drainage ; h initial = -11.48 cm (régime permanent, K(h) = 10 cm/j).
  Solute transport : Crank–Nicolson (epsi 0.5), Galerkin ; bulk density 1.5 g/cm3, disp. longitudinale 10.0 cm,
               Dw = 1 cm2/j ; Kd = 0, pas de dégradation. Upper = Concentration flux BC, c = 1 pendant
               tPulse = 0.5 j puis 0 ; Lower = Zero concentration gradient.
  Profile : 121 noeuds (dz = 0.5 cm), 0 à -60 cm ; concentration initiale 0. Observation nodes 10, 20, 40, 60 cm.
Résultat attendu : courbe de percée à 60 cm (solute1.out : cvBot ; OBS_NODE.OUT : Conc) — comparer
à la solution analytique de l'ADE pour une injection brève (v = q/θ = 30.32 cm/j, D = 10.0·v + θ·Dw·tau).
