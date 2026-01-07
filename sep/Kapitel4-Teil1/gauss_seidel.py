"""
Gauss-Seidel Iterative Method (CORRECTED)
Formula: x^(k+1) = -(D+L)^(-1) R x^(k) + (D+L)^(-1) b
Or equivalently: (D+L)x^(k+1) = -R x^(k) + b
"""
import numpy as np

def print_matrix(matrix, title=""):
    """Print matrix in readable format"""
    if title:
        print(f"\n{title}")
    for row in matrix:
        print([f"{val:8.2f}" if abs(val) > 1e-10 else f"{'0':>8}" for val in row])

def gauss_seidel_iteration(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Gauss-Seidel method:
    (D+L)x^(k+1) = -R x^(k) + b
    or x^(k+1) = -(D+L)^(-1) R x^(k) + (D+L)^(-1) b
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if x0 is None:
        x0 = np.zeros(n)

    print("\n" + "="*70)
    print("GAUSS-SEIDEL ITERATION METHOD")
    print("="*70)

    print("\n--- ORIGINAL SYSTEM Ax = b ---")
    print_matrix(A)
    print(f"\nb = {b}")

    # Decompose A = D + L + R
    D = np.diag(np.diag(A))           # Diagonal matrix
    L = np.tril(A, -1)                # Strict lower triangular
    R = np.triu(A, 1)                 # Strict upper triangular (R in the formula)

    print("\n--- DECOMPOSITION A = D + L + R ---")
    print("\nD (diagonal):")
    print_matrix(D)
    print("\nL (strict lower):")
    print_matrix(L)
    print("\nR (strict upper):")
    print_matrix(R)

    print("\n--- GAUSS-SEIDEL FORMULA ---")
    print("(D+L)x^(k+1) = -R x^(k) + b")
    print("x^(k+1) = -(D+L)^(-1) R x^(k) + (D+L)^(-1) b")

    # Precompute (D+L) and its inverse
    DL = D + L
    DL_inv = np.linalg.inv(DL)

    print(f"\nInitial guess x^(0) = {x0}")

    x = x0.copy()

    for k in range(max_iter):
        print(f"\n{'='*70}")
        print(f"ITERATION {k+1}")
        print(f"{'='*70}")

        x_old = x.copy()

        # x^(k+1) = -(D+L)^(-1) R x^(k) + (D+L)^(-1) b
        x_new = -DL_inv @ R @ x + DL_inv @ b

        x = x_new

        print(f"x^({k+1}) = {x}")

        # Check convergence
        error = np.max(np.abs(x - x_old))
        print(f"Max error |x^({k+1}) - x^({k})| = {error:.2e}")

        if error < tol:
            print(f"\n✓ CONVERGED after {k+1} iterations")
            break

    else:
        print(f"\n⚠️  Did not converge after {max_iter} iterations")

    # Verification
    residual_final = np.abs(A @ x - b)
    print(f"\n--- FINAL VERIFICATION ---")
    print(f"Ax = {A @ x}")
    print(f"b  = {b}")
    print(f"Final residual |Ax - b|_∞ = {np.max(residual_final):.2e}")

    print(f"\nFinal solution x = {x}")
    return x


if __name__ == "__main__":
    # Example from the textbook (Beispiel 4.17)
    A = [
        [4, -1, 1],
        [-2, 5, 1],
        [1, -2, 5]
    ]

    b = [5, 11, 12]

    x = gauss_seidel_iteration(A, b, tol=1e-5, max_iter=10)
