"""
Jacobi Iterative Method
Solves Ax = b using the Jacobi iteration formula
"""
import numpy as np


def print_matrix(matrix, title=""):
    """Print matrix in readable format"""
    if title:
        print(f"\n{title}")
    for row in matrix:
        print([f"{val:8.2f}" if abs(val) > 1e-10 else f"{'0':>8}" for val in row])


def jacobi_iteration(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Jacobi method: x^(k+1) = D^-1 (b - (L+U)x^(k))

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        x: Solution vector
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if x0 is None:
        x0 = np.zeros(n)

    print("\n" + "=" * 70)
    print("JACOBI ITERATION METHOD")
    print("=" * 70)

    print("\n--- ORIGINAL SYSTEM Ax = b ---")
    print_matrix(A)
    print(f"\nb = {b}")

    # Decompose A = D + L + U
    D = np.diag(np.diag(A))  # Diagonal matrix
    L = np.tril(A, -1)  # Strict lower triangular
    U = np.triu(A, 1)  # Strict upper triangular

    print("\n--- DECOMPOSITION A = D + L + U ---")
    print("\nD (diagonal):")
    print_matrix(D)
    print("\nL (strict lower):")
    print_matrix(L)
    print("\nU (strict upper):")
    print_matrix(U)

    print("\n--- JACOBI FORMULA ---")
    print("x^(k+1) = D^-1 (b - (L+U)x^(k))")
    print(f"D^-1 = diag({1 / np.diag(D)})")

    # Precompute D^-1
    D_inv = np.diag(1 / np.diag(D))

    print(f"\nInitial guess x^(0) = {x0}")

    x = x0.copy()
    x_new = np.zeros(n)

    for k in range(max_iter):
        print(f"\n{'=' * 70}")
        print(f"ITERATION {k + 1}")
        print(f"{'=' * 70}")

        # x^(k+1) = D^-1 (b - (L+U)x^(k))
        residual = b - (L + U) @ x

        print(f"Residual r^(k) = b - (L+U)x^(k):")
        print(residual)

        x_new = D_inv @ residual

        print(f"x^(k+1) = D^-1 * r^(k):")
        print(x_new)

        # Check convergence
        error = np.max(np.abs(x_new - x))
        print(f"Max error |x^(k+1) - x^(k)| = {error:.2e}")

        x = x_new.copy()

        if error < tol:
            print(f"\n✓ CONVERGED after {k + 1} iterations")
            break

    else:
        print(f"\n⚠️  Did not converge after {max_iter} iterations")

    # Verification
    residual_final = np.abs(A @ x - b)
    print(f"\n--- FINAL VERIFICATION ---")
    print(f"Final residual |Ax - b|_∞ = {np.max(residual_final):.2e}")

    print(f"\nFinal solution x = {x}")
    return x


if __name__ == "__main__":
    A = [
        [4, -1, 1],
        [-2, 5, 1],
        [1, -2, 5]
    ]

    b = [5, 11, 12]

    x = jacobi_iteration(A, b, tol=1e-5, max_iter=5)
