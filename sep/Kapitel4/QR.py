import numpy as np

# Beispielmatrix A
A = np.array([[1., 2., -1.],
              [4., -2., 6.],
              [3., 1., 0.]])
m, n = A.shape
R = A.copy().astype(float)
Q = np.eye(m)
for k in range(n):
    # Teilvektor der k-ten Spalte ab Zeile k
    x = R[k:, k]
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        continue
    # Wahl von alpha für numerische Stabilität
    alpha = -np.sign(x[0]) * norm_x
    e1 = np.zeros_like(x)
    e1[0] = 1.0
    # Householder-Vektor v und Normierung
    v = x - alpha * e1
    v = v / np.linalg.norm(v)
    # Kleine Householder-Matrix
    Hk_small = np.eye(len(x)) - 2.0 * np.outer(v, v)
    # In die volle Dimension einbetten
    Hk = np.eye(m)
    Hk[k:, k:] = Hk_small
    # Update von R und Q
    R = Hk @ R
    Q = Q @ Hk.T  # Hk.T == Hk, aber so ist es explizit

print("A =\n", A)
print("Q =\n", Q)
print("R =\n", R)
print("Überprüfung A ≈ Q @ R:", np.allclose(A, Q @ R))
