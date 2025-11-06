import numpy as np
from Yanik_Michael_S06_Aufg2 import Yanik_Michael_S06_Aufg2

# System 1
A1 = np.array([[2.0, 1.0, -1.0],
               [-3.0, -1.0, 2.0],
               [-2.0, 1.0, 2.0]])
b1 = np.array([8.0, -11.0, -3.0])

# System 2
A2 = np.array([[1.0, 2.0, 3.0],
               [0.0, 1.0, 4.0],
               [5.0, 6.0, 0.0]])
b2 = np.array([14.0, 13.0, 23.0])

# Liste all Systems (A,b)
systems = [(A1, b1), (A2, b2)]

for idx, (A, b) in enumerate(systems, start=1):
    print(f"\n--- System {idx} ---")

    # Solution with our Gauss-Algorithmus
    U, detA, x_my = Yanik_Michael_S06_Aufg2(A, b)
    print("Obere Dreiecksmatrix U:\n", U)
    print("Determinante det(A):", detA)
    print("Lösung x (eigener Algorithmus):", x_my)

    # Solution with numpy.linalg.solve
    x_np = np.linalg.solve(A, b)
    print("Lösung x (numpy.linalg.solve):", x_np)

    # Check differences
    diff = np.abs(x_my - x_np)
    print("Abweichung |x_my - x_np|:", diff)

# The solutions agree except for small rounding differences.
# Deviations are due to numerical rounding errors in the Gaussian method.
# Mathematically, the results are identical.