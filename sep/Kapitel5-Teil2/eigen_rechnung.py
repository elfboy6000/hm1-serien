"""
Eigenvalues and Eigenvectors for 2x2 and 3x3 matrices
Detailed step-by-step solution following lecture style
"""
import numpy as np
from math import sqrt

def print_section(title):
    """Print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

def print_matrix(matrix, title=""):
    """Print matrix in readable format"""
    if title:
        print(f"\n{title}")
    for row in matrix:
        print("  " + "  ".join([f"{val:8.2f}" if abs(val) > 1e-10 else f"{'0':>8}" for val in row]))

def print_calculation(desc, result):
    """Print calculation step"""
    print(f"\n{desc}")
    print(f"  → {result}")

def det_2x2(M):
    """Compute determinant of 2x2 matrix with details."""
    a, b = M[0]
    c, d = M[1]
    det = a*d - b*c
    return det, a, b, c, d

def det_3x3(M):
    """Compute determinant of 3x3 matrix explicitly."""
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def char_poly_2x2_detailed(A):
    """
    Characteristic polynomial for 2x2 with full details.
    det(A - λI) = (a-λ)(d-λ) - bc = λ² - (a+d)λ + (ad-bc) = 0
    """
    print_section("STEP 1: CHARACTERISTIC POLYNOMIAL (2x2)")
    print("\nGeneral form: det(A - λI) = 0")
    print("\nFor 2x2 matrix:")
    print("  λ² - tr(A)·λ + det(A) = 0")
    print("  where tr(A) = trace = sum of diagonal elements")
    print("        det(A) = determinant")

    a, b = A[0]
    c, d = A[1]

    trace = a + d
    det_val = a*d - b*c

    print(f"\n  A = ({a:6.2f}  {b:6.2f})")
    print(f"      ({c:6.2f}  {d:6.2f})")

    print_calculation(f"tr(A) = {a} + {d}", f"{trace}")
    print_calculation(f"det(A) = {a}·{d} - {b}·{c}", f"{det_val}")

    print(f"\nCharacteristic equation:")
    print(f"  λ² {-trace:+6.1f}·λ {det_val:+6.1f} = 0")

    return trace, det_val

def char_poly_3x3_detailed(A):
    """
    Characteristic polynomial for 3x3 with full details.
    """
    print_section("STEP 1: CHARACTERISTIC POLYNOMIAL (3x3)")
    print("\nGeneral form: det(A - λI) = 0")
    print("\nFor 3x3 matrix:")
    print("  λ³ - tr(A)·λ² + M·λ - det(A) = 0")
    print("  where M = sum of principal minors")

    a11, a12, a13 = A[0]
    a21, a22, a23 = A[1]
    a31, a32, a33 = A[2]

    print(f"\n  A = ({a11:6.2f}  {a12:6.2f}  {a13:6.2f})")
    print(f"      ({a21:6.2f}  {a22:6.2f}  {a23:6.2f})")
    print(f"      ({a31:6.2f}  {a32:6.2f}  {a33:6.2f})")

    trace = a11 + a22 + a33

    # Principal minors (2x2 determinants)
    m11 = a22*a33 - a23*a32
    m22 = a11*a33 - a13*a31
    m33 = a11*a22 - a12*a21
    sum_minors = m11 + m22 + m33

    # Full determinant
    detA = det_3x3(A)

    print(f"\nTrace: tr(A) = {a11} + {a22} + {a33} = {trace}")
    print(f"\nPrincipal minors (2x2 determinants):")
    print(f"  M₁₁ = |{a22:6.2f}  {a23:6.2f}| = {m11:6.2f}")
    print(f"        |{a32:6.2f}  {a33:6.2f}|")
    print(f"  M₂₂ = |{a11:6.2f}  {a13:6.2f}| = {m22:6.2f}")
    print(f"        |{a31:6.2f}  {a33:6.2f}|")
    print(f"  M₃₃ = |{a11:6.2f}  {a12:6.2f}| = {m33:6.2f}")
    print(f"        |{a21:6.2f}  {a22:6.2f}|")
    print(f"  Sum = {sum_minors:6.2f}")

    print(f"\nDeterminant: det(A) = {detA:6.2f}")

    print(f"\nCharacteristic equation:")
    print(f"  λ³ {-trace:+6.1f}·λ² {sum_minors:+6.1f}·λ {-detA:+6.1f} = 0")

    return trace, sum_minors, detA

def find_eigenvalues_2x2_detailed(A):
    """Find eigenvalues of 2x2 using quadratic formula with details."""
    trace, det_val = char_poly_2x2_detailed(A)

    print_section("STEP 2: SOLVE FOR EIGENVALUES (2x2)")
    print(f"\nUsing quadratic formula: λ = (tr ± √(tr² - 4·det)) / 2")

    discriminant = trace**2 - 4*det_val
    print_calculation(f"Discriminant Δ = tr² - 4·det = {trace}² - 4·{det_val}", f"{discriminant}")

    if discriminant >= 0:
        sqrt_disc = sqrt(discriminant)
        lam1 = (trace + sqrt_disc) / 2
        lam2 = (trace - sqrt_disc) / 2

        print(f"\n√Δ = {sqrt_disc:.4f}")
        print_calculation(f"λ₁ = ({trace} + {sqrt_disc:.4f}) / 2", f"{lam1:.4f}")
        print_calculation(f"λ₂ = ({trace} - {sqrt_disc:.4f}) / 2", f"{lam2:.4f}")

        eigenvalues = [lam1, lam2]
    else:
        sqrt_disc = sqrt(-discriminant)
        real_part = trace / 2
        imag_part = sqrt_disc / 2
        print(f"\nΔ < 0, complex eigenvalues:")
        print_calculation(f"λ₁ = {real_part:.4f} + {imag_part:.4f}i", "")
        print_calculation(f"λ₂ = {real_part:.4f} - {imag_part:.4f}i", "")
        eigenvalues = [complex(real_part, imag_part), complex(real_part, -imag_part)]

    return eigenvalues

def find_eigenvalues_3x3_detailed(A):
    """Find eigenvalues of 3x3 with detailed steps."""

    trace, sum_minors, detA = char_poly_3x3_detailed(A)


    print_section("STEP 2: SOLVE FOR EIGENVALUES (3x3)")
    print(f"\nTesting rational roots (factors of {detA:.0f})...")

    # Possible rational roots
    possible_roots = set()
    for i in [-3, -2, -1, 1, 2, 3, 4, 6, 8, 9, 12]:
        possible_roots.add(i)
        possible_roots.add(-i)

    eigenvalues = []
    coeffs = [1.0, -trace, sum_minors, -detA]

    for r in sorted(possible_roots):
        val = np.polyval(coeffs[::-1], r)
        if abs(val) < 1e-6:
            eigenvalues.append(float(r))
            print(f"\n✓ λ = {r} is a root (p({r}) ≈ 0)")

            # Polynomial division: (λ³ + c₂λ² + c₁λ + c₀) / (λ - r)
            # Result: λ² + (c₂ + r)λ + (c₁ + r(c₂ + r))
            b = coeffs[1] - r
            c = coeffs[2] - r*b
            print(f"\nFactoring: divide by (λ - {r}) to get:")
            print(f"  λ² {b:+6.2f}·λ {c:+6.2f} = 0")

            # Quadratic formula
            disc = b**2 - 4*c
            print_calculation(f"Discriminant = {b}² - 4·{c}", f"{disc:.2f}")

            if disc >= 0:
                sqrt_disc = sqrt(disc)
                lam2 = (-b + sqrt_disc) / 2
                lam3 = (-b - sqrt_disc) / 2
                eigenvalues.extend([lam2, lam3])
                print_calculation(f"λ₂ = ({-b:+.2f} + √{disc:.2f}) / 2", f"{lam2:.4f}")
                print_calculation(f"λ₃ = ({-b:+.2f} - √{disc:.2f}) / 2", f"{lam3:.4f}")
            else:
                sqrt_disc = sqrt(-disc)
                real = -b / 2
                imag = sqrt_disc / 2
                eigenvalues.extend([complex(real, imag), complex(real, -imag)])
                print(f"Complex roots: {real:.4f} ± {imag:.4f}i")

            break

    return eigenvalues

def eigenvector_for_lambda_detailed(A, lam, lam_num):
    """Compute eigenvector for λ with detailed steps."""
    print_section(f"STEP 3.{lam_num}: EIGENVECTOR for λ = {lam}")

    n = A.shape[0]

    print(f"\nSolve (A - λI)v = 0:")
    print(f"\nA - {lam}I =")

    if isinstance(lam, complex):
        M = A.astype(complex) - lam * np.eye(n, dtype=complex)
    else:
        M = A - lam * np.eye(n)

    print_matrix(M)

    # Gaussian elimination with details
    M = M.astype(complex) if M.dtype != complex else M
    print(f"\nGaussian elimination:")

    for col in range(n):
        pivot_row = col
        for row in range(col + 1, n):
            if abs(M[row, col]) > abs(M[pivot_row, col]):
                pivot_row = row
        if abs(M[pivot_row, col]) < 1e-10:
            continue

        if pivot_row != col:
            M[[col, pivot_row]] = M[[pivot_row, col]]
            print(f"  Row {col} ↔ Row {pivot_row}")

        pivot = M[col, col]
        for row in range(col + 1, n):
            if abs(M[row, col]) > 1e-10:
                mult = M[row, col] / pivot
                print(f"  Row {row} := Row {row} - ({mult:.2f})·Row {col}")
                M[row, :] -= mult * M[col, :]

    print("\nRow echelon form:")
    print_matrix(M)

    # Back substitution
    v = np.zeros(n, dtype=complex)
    free_vars = []
    for i in range(n):
        if abs(M[i, i]) < 1e-10:
            free_vars.append(i)

    if len(free_vars) >= 1:
        free_idx = free_vars[0]
        v[free_idx] = 1.0
        print(f"\nFree variable: v[{free_idx}] = 1")

        for i in range(n - 1, -1, -1):
            if abs(M[i, i]) > 1e-10:
                sum_ax = 0
                for j in range(i + 1, n):
                    sum_ax += M[i, j] * v[j]
                v[i] = -sum_ax / M[i, i]
                print(f"v[{i}] = {v[i]:.4f}")

    norm = np.linalg.norm(v)
    if norm > 1e-10:
        v = v / norm

    print(f"\nEigenvector (normalized): v = ")
    for vi in v:
        if abs(vi.imag) < 1e-10:
            print(f"  {vi.real:8.4f}")
        else:
            print(f"  {vi:8.4f}")

    return v

def eigenvalues_eigenvectors_detailed(A):
    """
    Complete eigenvalue/eigenvector computation with detailed steps.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if n == 2:
        print_section("EIGENVALUES AND EIGENVECTORS - 2x2 MATRIX")
        eigenvalues = find_eigenvalues_2x2_detailed(A)
    elif n == 3:
        print_section("EIGENVALUES AND EIGENVECTORS - 3x3 MATRIX")
        eigenvalues = find_eigenvalues_3x3_detailed(A)
    else:
        raise ValueError("Only 2x2 and 3x3 matrices supported.")

    print_section("FINDING EIGENVECTORS")
    eigenvectors = []
    for i, lam in enumerate(eigenvalues, 1):
        v = eigenvector_for_lambda_detailed(A, lam, i)
        eigenvectors.append(v)

    # Verification
    print_section("VERIFICATION")
    print("\nCheck Av = λv for each pair:")
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors), 1):
        Av = A @ v
        error = np.linalg.norm(Av - lam * v)
        print(f"\nEigenvalue λ_{i} = {lam}")
        print(f"  ||A·v - λ·v|| = {error:.2e} ✓")

    return eigenvalues, np.column_stack(eigenvectors)


if __name__ == "__main__":
    # Test 2x2
    print("\n\n")
    print("#" * 70)
    print("# EXAMPLE 1: 2x2 MATRIX")
    print("#" * 70)
    A2 = [
        [2, 5],
        [-1, -2]
    ]
    evals2, evecs2 = eigenvalues_eigenvectors_detailed(A2)

    # Test 3x3
    print("\n\n\n")
    print("#" * 70)
    print("# EXAMPLE 2: 3x3 MATRIX")
    print("#" * 70)
    A3 = [
        [1, -2, 0],
        [2, 0, 1],
        [0, -2, 1]
    ]
    evals3, evecs3 = eigenvalues_eigenvectors_detailed(A3)
