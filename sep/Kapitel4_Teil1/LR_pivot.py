"""
LR decomposition with column pivoting + solve Ax = b
Compatible style with gaussian_elimination_classic
PA = L R  (P permutation matrix)
"""
import numpy as np


def print_matrix(matrix, title=""):
    """Print matrix in readable format"""
    if title:
        print(f"\n{title}")
    for row in matrix:
        print([f"{val:8.2f}" if abs(val) > 1e-10 else f"{'0':>8}" for val in row])


def lr_decomposition_pivot(A):
    """
    Compute P, L, R such that P A = L R using Doolittle with column pivoting.
    L has 1 on diagonal. A is not modified.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Start with identity permutation matrix
    P = np.eye(n)
    L = np.eye(n)
    R = A.copy()

    for col in range(n - 1):
        # Spaltenpivotisierung: größtes |R[row, col]| für row >= col wählen
        pivot_row = col + np.argmax(np.abs(R[col:, col]))
        if abs(R[pivot_row, col]) < 1e-10:
            raise ValueError(f"Zero column (or very small pivot) in column {col}, matrix not regular.")

        # Zeilen in R vertauschen
        if pivot_row != col:
            R[[col, pivot_row], :] = R[[pivot_row, col], :]
            P[[col, pivot_row], :] = P[[pivot_row, col], :]

            # Auch schon berechnete L-Einträge unterhalb der Diagonale mit vertauschen
            if col > 0:
                L[[col, pivot_row], :col] = L[[pivot_row, col], :col]

        pivot = R[col, col]

        # Vorwärts-Elimination, Multiplikatoren in L speichern
        for row in range(col + 1, n):
            multiplier = R[row, col] / pivot
            L[row, col] = multiplier
            R[row, col:] = R[row, col:] - multiplier * R[col, col:]

    print("\n--- PERMUTATION MATRIX P ---")
    print_matrix(P)
    print("\n--- MATRIX L ---")
    print_matrix(L)
    print("\n--- MATRIX R (upper) ---")
    print_matrix(R)

    return P, L, R


def forward_substitution(L, b):
    """Solve Ly = b with L lower-triangular (diag = 1)."""
    n = len(b)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        # L[i,i] = 1, so no division

    print(f"\nIntermediate vector y (Ly = b): {y}")
    return y


def back_substitution(R, y):
    """Solve Rx = y with R upper-triangular."""
    n = len(y)
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= R[i, j] * x[j]
        coeff = R[i, i]
        if abs(coeff) < 1e-10:
            raise ValueError(f"Zero (or very small) diagonal element R[{i},{i}] during back substitution.")
        x[i] = x[i] / coeff

    return x


def det_from_R(R, P) -> float:
    """
    det(A) = det(P)^(-1) * product(diagonal of R), since P A = L R and det(L)=1.
    det(P) is +/-1 depending on number of row swaps.
    """
    # Anzahl Vertauschungen aus Permutationsmatrix P bestimmen
    # (Permutation aus P auslesen und Signum bestimmen)
    perm = np.argmax(P, axis=1)
    visited = np.zeros_like(perm, dtype=bool)
    swaps = 0
    for i in range(len(perm)):
        if not visited[i]:
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len > 0:
                swaps += cycle_len - 1
    sign = -1 if swaps % 2 == 1 else 1

    determinant_R = 1.0
    for i in range(R.shape[0]):
        determinant_R *= R[i, i]

    # det(P A) = det(P) * det(A) = det(R)
    # => det(A) = det(R) / det(P) = sign * det(R) (da det(P) = sign)
    return determinant_R / sign


def lr_solve_pivot(A, b):
    """
    Solve Ax = b via LR decomposition with column pivoting:
      1) P A = L R
      2) Ly = P b (forward)
      3) R x = y (backward)
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    print("\n--- ORIGINAL MATRIX A ---")
    print_matrix(A)
    print("\n--- VECTOR b ---")
    print(b)

    # 1) LR decomposition with pivoting
    P, L, R = lr_decomposition_pivot(A)

    # determinant from R and P
    determinant = det_from_R(R, P)
    print(f"\nDeterminant of A (from P and R): {determinant:.2f}")

    # 2) Ly = P b  (Permutation auf rechte Seite anwenden)
    Pb = P @ b
    print(f"\nPermuted right-hand side Pb: {Pb}")
    y = forward_substitution(L, Pb)

    # 3) Rx = y
    x = back_substitution(R, y)

    print(f"\nSolution x = {x}")
    return x


if __name__ == "__main__":
    A = [
        [3, 9, 12, 12],
        [-2, -5, 7, 2],
        [6, 12, 18, 6],
        [3, 7, 38, 14]
    ]

    b = [51, 2, 54, 79]

    x = lr_solve_pivot(A, b)
