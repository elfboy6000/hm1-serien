"""
Convergence Analysis for Iterative Methods
Implements Banach Fixed Point Theorem and error bounds
x^(n+1) = B x^(n) + c = F(x^(n))

Convergence criteria:
- Converging fixed point: ||B|| < 1
- Diverging fixed point: ||B|| > 1

Error bounds:
- A-priori: ||x^(n) - x̄|| <= (||B||^n / (1 - ||B||)) ||x^(1) - x^(0)||
- A-posteriori: ||x^(n) - x̄|| <= (||B|| / (1 - ||B||)) ||x^(n) - x^(n-1)||
"""
import numpy as np


def print_matrix(matrix, title=""):
    """Print matrix in readable format"""
    if title:
        print(f"\n{title}")
    for row in matrix:
        print([f"{val:8.4f}" if abs(val) > 1e-10 else f"{'0':>8}" for val in row])


def convergence_analysis(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Analyze convergence of iterative methods.
    Decomposes A = D + L + R and computes iteration matrix B.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if x0 is None:
        x0 = np.zeros(n)

    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS FOR ITERATIVE METHODS")
    print("=" * 70)

    print("\n--- ORIGINAL SYSTEM Ax = b ---")
    print_matrix(A)
    print(f"\nb = {b}")

    # Decompose A = D + L + R
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    R = np.triu(A, 1)

    print("\n--- DECOMPOSITION A = D + L + R ---")
    print("\nD (diagonal):")
    print_matrix(D)
    print("\nL (strict lower):")
    print_matrix(L)
    print("\nR (strict upper):")
    print_matrix(R)

    # ==========================================
    # JACOBI METHOD ANALYSIS
    # ==========================================
    print("\n" + "=" * 70)
    print("JACOBI METHOD: x^(k+1) = -D^(-1)(L+R)x^(k) + D^(-1)b")
    print("=" * 70)

    D_inv = np.linalg.inv(D)
    B_jacobi = -D_inv @ (L + R)
    c_jacobi = D_inv @ b

    print("\nIteration matrix B (Jacobi):")
    print_matrix(B_jacobi)
    print(f"\nVector c (Jacobi): {c_jacobi}")

    # Compute spectral radius and norms
    eigenvalues_jacobi = np.linalg.eigvals(B_jacobi)
    spectral_radius_jacobi = np.max(np.abs(eigenvalues_jacobi))
    norm_jacobi = np.linalg.norm(B_jacobi, ord=np.inf)  # Infinity norm

    print(f"\nEigenvalues of B (Jacobi): {eigenvalues_jacobi}")
    print(f"Spectral radius ρ(B) = {spectral_radius_jacobi:.6f}")
    print(f"Infinity norm ||B||_∞ = {norm_jacobi:.6f}")

    if spectral_radius_jacobi < 1:
        print(f"✓ JACOBI CONVERGES (ρ(B) < 1)")
    else:
        print(f"✗ JACOBI DIVERGES (ρ(B) >= 1)")

    # ==========================================
    # GAUSS-SEIDEL METHOD ANALYSIS
    # ==========================================
    print("\n" + "=" * 70)
    print("GAUSS-SEIDEL METHOD: x^(k+1) = -(D+L)^(-1)Rx^(k) + (D+L)^(-1)b")
    print("=" * 70)

    DL = D + L
    DL_inv = np.linalg.inv(DL)
    B_gs = -DL_inv @ R
    c_gs = DL_inv @ b

    print("\nIteration matrix B (Gauss-Seidel):")
    print_matrix(B_gs)
    print(f"\nVector c (Gauss-Seidel): {c_gs}")

    eigenvalues_gs = np.linalg.eigvals(B_gs)
    spectral_radius_gs = np.max(np.abs(eigenvalues_gs))
    norm_gs = np.linalg.norm(B_gs, ord=np.inf)

    print(f"\nEigenvalues of B (Gauss-Seidel): {eigenvalues_gs}")
    print(f"Spectral radius ρ(B) = {spectral_radius_gs:.6f}")
    print(f"Infinity norm ||B||_∞ = {norm_gs:.6f}")

    if spectral_radius_gs < 1:
        print(f"✓ GAUSS-SEIDEL CONVERGES (ρ(B) < 1)")
    else:
        print(f"✗ GAUSS-SEIDEL DIVERGES (ρ(B) >= 1)")

    # ==========================================
    # COMPARISON
    # ==========================================
    print("\n" + "=" * 70)
    print("CONVERGENCE COMPARISON")
    print("=" * 70)

    print(f"\nJacobi vs Gauss-Seidel:")
    print(f"  Jacobi spectral radius:      {spectral_radius_jacobi:.6f}")
    print(f"  Gauss-Seidel spectral radius: {spectral_radius_gs:.6f}")

    if spectral_radius_gs < spectral_radius_jacobi:
        print(f"  → Gauss-Seidel converges faster")
    elif spectral_radius_jacobi < spectral_radius_gs:
        print(f"  → Jacobi converges faster")
    else:
        print(f"  → Same convergence rate")

    # ==========================================
    # RUN JACOBI WITH ERROR BOUNDS
    # ==========================================
    print("\n" + "=" * 70)
    print("JACOBI ITERATION WITH ERROR BOUNDS")
    print("=" * 70)
    print(f"\nStarting point: x^(0) = {x0}")

    x = x0.copy()
    x_prev = x.copy()

    # Compute exact solution (for comparison)
    x_exact = np.linalg.solve(A, b)
    print(f"Exact solution: x̄ = {x_exact}")

    for k in range(max_iter):
        x_prev = x.copy()
        x = B_jacobi @ x + c_jacobi

        # Compute error
        error_actual = np.linalg.norm(x - x_exact, ord=np.inf)
        error_diff = np.linalg.norm(x - x_prev, ord=np.inf)

        # A-posteriori bound
        if spectral_radius_jacobi < 1:
            error_aposteriori = (spectral_radius_jacobi / (1 - spectral_radius_jacobi)) * error_diff
        else:
            error_aposteriori = float('inf')

        print(f"\nIteration {k + 1}:")
        print(f"  x^({k + 1}) = {x}")
        print(f"  ||x^({k + 1}) - x^({k})||_∞ = {error_diff:.2e}")
        print(f"  ||x^({k + 1}) - x̄||_∞ (actual) = {error_actual:.2e}")
        print(f"  ||x^({k + 1}) - x̄||_∞ (a-posteriori bound) = {error_aposteriori:.2e}")

        if error_actual < tol:
            print(f"\n✓ CONVERGED after {k + 1} iterations")
            break

    print(f"\n--- FINAL VERIFICATION ---")
    print(f"Ax = {A @ x}")
    print(f"b  = {b}")
    print(f"Final residual |Ax - b|_∞ = {np.max(np.abs(A @ x - b)):.2e}")

    return x, spectral_radius_jacobi, spectral_radius_gs


if __name__ == "__main__":
    A = [
        [4, -1, 1],
        [-2, 5, 1],
        [1, -2, 5]
    ]

    b = [5, 11, 12]

    x, rho_jacobi, rho_gs = convergence_analysis(A, b, tol=1e-5, max_iter=15)
