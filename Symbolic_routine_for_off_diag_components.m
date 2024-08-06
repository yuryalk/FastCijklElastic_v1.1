syms C12 C13 C23 M_E22 M_E11 M_E33 M_Sdev11 M_Sdev22 M_Sdev33
eqn1 = M_E22*C12 + M_E33*C13 == M_Sdev11;
eqn2 = M_E11*C12 + M_E33*C23 == M_Sdev22;
eqn3 = M_E11*C13 + M_E22*C23 == M_Sdev33;

sol = solve([eqn1, eqn2, eqn3], [C12, C13, C23]);
C12s = sol.C12
C13s = sol.C13
C23s = sol.C23