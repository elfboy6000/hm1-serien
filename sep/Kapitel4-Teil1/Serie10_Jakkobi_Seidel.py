from enum import Enum
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import shutil
import time


class IterMethod(Enum):
    JACOBI = "jacobi"
    GAUSS_SEIDEL = "gauss_seidel"


def solveMatrixIter(
        A: NDArray[np.float64],  # Koeffizientenmatrix (n x n)
        b: NDArray[np.float64],  # Rechte Seite (n x 1)
        x0: NDArray[np.float64],  # Startvektor (n x 1)
        tol: float,  # Fehlertoleranz für Abbruch
        opt: IterMethod,  # "jacobi" oder "gauss_seidel"
) -> tuple[NDArray[np.float64], int, int]:
    """
    Solve the linear system Ax = b using the specified iterative method.
    """

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    R = np.triu(A, 1)

    if opt == IterMethod.JACOBI:
        B = -np.linalg.inv(D) @ (L + R)
    elif opt == IterMethod.GAUSS_SEIDEL:
        B = -np.linalg.inv(D + L) @ R
    else:
        raise ValueError("Invalid iteration method option.")

    x1 = (
        B @ x0 + np.linalg.inv(D + L) @ b
        if opt == IterMethod.GAUSS_SEIDEL
        else B @ x0 + np.linalg.inv(D) @ b
    )

    B_norm = np.linalg.norm(B, np.inf)
    if B_norm >= 1:
        raise ValueError("The method does not converge (spectral radius >= 1).")
    delta_x = np.linalg.norm(x1 - x0, np.inf)
    n2 = calc_num_iterations(tol, B_norm, delta_x)

    if opt == IterMethod.JACOBI:
        result, num_iterations = calculate_jacobi(L, R, D, b, x0, tol)
    elif opt == IterMethod.GAUSS_SEIDEL:
        result, num_iterations = calculate_seidel(b, D, L, R, x0, tol)
    else:
        raise ValueError("Invalid iteration method option.")

    return result, num_iterations, n2


def calc_num_iterations(tol, B_norm, delta_x) -> int:
    """
    Calculates the a-priori number of iterations needed for convergence.

    Returns: int: Estimated number of iterations
    """
    res = np.log(tol * (1 - B_norm) / delta_x) / np.log(B_norm)
    return int(np.ceil(res))


def calculate_jacobi(
        L: NDArray[np.float64],
        R: NDArray[np.float64],
        D: NDArray[np.float64],
        b: NDArray[np.float64],
        x0: NDArray[np.float64],
        tol: float,
) -> tuple[NDArray[np.float64], int]:
    D_inv = np.linalg.inv(D)

    counter = 0
    x_current = x0.copy()
    x_next = np.zeros_like(x0)

    while True:
        x_next = D_inv @ (b - (L + R) @ x_current)
        counter += 1
        if np.linalg.norm(x_next - x_current, np.inf) < tol:
            break
        x_current = x_next
    return x_next, counter


def calculate_seidel(
        b: NDArray[np.float64],
        D: NDArray[np.float64],
        L: NDArray[np.float64],
        R: NDArray[np.float64],
        x0: NDArray[np.float64],
        tol: float,
) -> tuple[NDArray[np.float64], int]:
    D_L_inv = np.linalg.inv(D + L)
    x_current = x0.copy()

    counter = 0
    x_next = np.zeros_like(x0)

    while True:
        counter += 1
        x_next = -D_L_inv @ R @ x_current + D_L_inv @ b

        if np.linalg.norm(x_next - x_current, np.inf) < tol:
            break

        x_current = x_next
    return x_next, counter


def benchmark() -> None:
    """
    Benchmark-Script to compare performance of Jacobi, Gauss-Seidel and NumPy's built-in solver.
    """
    dim = 1000

    A = np.diag(np.diag(np.ones((dim, dim)) * 4000)) + np.ones((dim, dim))
    dum1 = np.arange(1, int(dim / 2 + 1), dtype=np.float64).reshape((int(dim / 2), 1))
    dum2 = np.arange(int(dim / 2), 0, -1, dtype=np.float64).reshape((int(dim / 2), 1))

    x = np.append(dum1, dum2, axis=0)
    b = A @ x
    x0 = np.zeros((dim, 1))
    tol = 1e-4

    start_jacobi = time.time()
    x_jacobi, _, _ = solveMatrixIter(A, b, x0, tol, IterMethod.JACOBI)
    end_jacobi = time.time()
    jacobi_time = end_jacobi - start_jacobi

    start_seidel = time.time()
    x_seidel, _, _ = solveMatrixIter(A, b, x0, tol, IterMethod.GAUSS_SEIDEL)
    end_seidel = time.time()
    seidel_time = end_seidel - start_seidel

    np_solve_start = time.time()
    x_numpy = np.linalg.solve(A, b)
    np_solve_end = time.time()
    np_solve_time = np_solve_end - np_solve_start

    print(f"Jacobi Method Time: {jacobi_time:.4f} seconds")
    print(f"Gauss-Seidel Method Time: {seidel_time:.4f} seconds")
    print(f"NumPy Solve Time: {np_solve_time:.4f} seconds")

    # Berechne absolute Fehler für jede Methode
    error_jacobi = np.abs(x_jacobi - x)
    error_seidel = np.abs(x_seidel - x)
    error_numpy = np.abs(x_numpy - x)

    # Plot der absoluten Fehler
    plt.figure(figsize=(12, 6))
    plt.semilogy(error_numpy, label='NumPy Solve', linewidth=2, alpha=0.7)
    plt.semilogy(error_jacobi, label='Jacobi', linewidth=2, alpha=0.7)
    plt.semilogy(error_seidel, label='Gauss-Seidel', linewidth=2, alpha=0.7)

    plt.xlabel('Vektorindex')
    plt.ylabel('Absoluter Fehler (log-Skala)')
    plt.title('Absoluter Fehler der Lösungsvektoren')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    """
    Beobachtungen:

    1. NumPy's Löser hat den kleinsten Fehler (~1e-10 bis 1e-12), da es direkte
       Verfahren (LU-Zerlegung) verwendet statt iterativer Methoden.

    2. Jacobi und Gauss-Seidel zeigen vergleichbare Fehler (~1e-4 bis 1e-3),
       was der gewählten Toleranz (tol=1e-4) entspricht.

    3. Die Fehler sind nicht uniform über den Vektor verteilt - es gibt
       Variationen, die auf die Struktur der Matrix A hinweisen.

    4. Trotz der höheren Genauigkeit ist NumPy's Löser auch deutlich schneller
       (~0.006s vs ~0.09-0.12s), was zeigt, dass direkte Verfahren für diese
       Problemgrösse effizienter sind als iterative Methoden.
    """


def print_result(opt: IterMethod, xn: NDArray[np.float64], n: int, n2: int) -> None:
    method = "Jacobi" if opt == IterMethod.JACOBI else "Gauss-Seidel"
    xn_flat = np.asarray(xn).flatten()
    width = shutil.get_terminal_size((80, 20)).columns
    sep = "-" * min(width, 80)

    print(sep)
    print(f"Method: {method}")
    print(f"Iterations (actual): {n}")
    print(f"Iterations (a-priori estimate): {n2}")
    print("Solution vector (first 10 entries shown if long):")
    to_show = xn_flat if xn_flat.size <= 10 else xn_flat[:10]
    for i, val in enumerate(to_show, start=1):
        print(f" x[{i:>2}] = {val:.6f}")
    if xn_flat.size > 10:
        print(f" ... ({xn_flat.size - 10} more entries)")
    print(sep)


if __name__ == "__main__":
    A = np.array([[7, -2, -2], [-2, 7, -2], [-2, 7, -2]])
    b = np.array([5, -13, 14])
    x0 = np.array([0, 0, 0])
    tol = 1e-9

    xn_jacobi, n_jacobi, n2_jacobi = solveMatrixIter(
        A, b, x0, tol=tol, opt=IterMethod.JACOBI
    )

    print_result(IterMethod.JACOBI, xn_jacobi, n_jacobi, n2_jacobi)

    xn_seidel, n_seidel, n2_seidel = solveMatrixIter(
        A, b, x0, tol=tol, opt=IterMethod.GAUSS_SEIDEL
    )

    print("---")

    print_result(IterMethod.GAUSS_SEIDEL, xn_seidel, n_seidel, n2_seidel)

    print("---")

    benchmark()

"""
Jacobi Method Time: 3.8359 seconds
Gauss-Seidel Method Time: 6.9979 seconds
NumPy Solve Time: 0.0203 seconds

-------------------------------

Unsere Implementationen sind einiges langsamer als NumPy's eingebaute Löseroutine.
"""
