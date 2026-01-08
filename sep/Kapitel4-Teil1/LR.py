"""
LR decomposition + solve Ax = b
Compatible style with gaussian_elimination_classic
"""
import numpy as np

def print_matrix(matrix, title=""):
    """Print matrix in readable format"""
    if title:
        print(f"\n{title}")
    for row in matrix:
        print([f"{val:8.2f}" if abs(val) > 1e-10 else f"{'0':>8}" for val in row])

def lr_decomposition(A):
    """
    Compute L and R (U) such that A = L R using Doolittle (L has 1 on diagonal).
    A is not modified.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    L = np.eye(n)
    R = A.copy()

    # Forward elimination, but store multipliers in L instead of changing b
    for col in range(n - 1):
        pivot = R[col, col]
        if abs(pivot) < 1e-10:
            raise ValueError(f"Zero pivot at position ({col}, {col}). Cannot continue (needs pivoting).")

        for row in range(col + 1, n):
            multiplier = R[row, col] / pivot
            L[row, col] = multiplier
            R[row, col:] = R[row, col:] - multiplier * R[col, col:]

    print("\n--- MATRIX L ---")
    print_matrix(L)
    print("\n--- MATRIX R (upper) ---")
    print_matrix(R)

    return L, R

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
        x[i] = x[i] / coeff

    return x

def det_from_R(R) -> float:
    """Det(A) = product of diagonal of R, since A = L R and det(L)=1."""
    determinant = 1.0
    for i in range(R.shape[0]):
        determinant *= R[i, i]
    return determinant

def lr_solve(A, b):
    """
    Solve Ax = b via LR decomposition:
      1) A = L R
      2) Ly = b (forward)
      3) Rx = y (backward)
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    print("\n--- ORIGINAL MATRIX A ---")
    print_matrix(A)
    print("\n--- VECTOR b ---")
    print(b)

    # 1) LR decomposition
    L, R = lr_decomposition(A)

    # determinant from R
    determinant = det_from_R(R)
    print(f"\nDeterminant of A (from R): {determinant:.2f}")

    # 2) Ly = b
    y = forward_substitution(L, b)

    # 3) Rx = y
    x = back_substitution(R, y)

    print(f"\nSolution x = {x}")
    return x


if __name__ == "__main__":
    A = [
        [-1, 1, 1],
        [1, -3, -2],
        [5, 1, 4]
    ]

    b = [0, 5, 3]

    x = lr_solve(A, b)
