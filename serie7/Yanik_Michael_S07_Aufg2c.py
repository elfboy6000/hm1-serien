import numpy as np
from scipy.linalg import lu

A = np.array([[0.8, 2.2, 3.6],
              [2.0, 3.0, 4.0],
              [1.2, 2.0, 5.8]], dtype=float)

b = np.array([2.4, 1.0, 4.0], dtype=float)

P, L, U = lu(A)

print("P =\n", P)
print("L =\n", L)
print("U =\n", U)

print("\nReconstruction check (P @ L @ U == A):", np.allclose(P @ L @ U, A))
print("Difference:\n", (P @ L @ U) - A)

# Lösung des Systems
x = np.linalg.solve(A, b)
print("\nLösung x =", x)
