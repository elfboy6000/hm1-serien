"""
A-Priori Error Estimation
Based on Banach Fixed Point Theorem

For iterative method x^(n+1) = F(x^(n)) with contraction constant q:
Error bound: ||x^(n) - x̄|| <= (q^n / (1 - q)) * ||x^(1) - x^(0)||

where:
- x^(n) is the approximation after n iterations
- x̄ is the exact solution (fixed point)
- q is the contraction constant (e.g., ||B|| for linear iteration)
- ||x^(1) - x^(0)|| is the initial step size
"""
import numpy as np


def a_priori_error_bound(n_iterations, contraction_constant, initial_step_norm):
    """
    Compute a-priori error bound for iterative methods.

    Formula: ||x^(n) - x̄|| <= (q^n / (1 - q)) * ||x^(1) - x^(0)||

    This bound can be computed BEFORE running iterations if we know:
    - The contraction constant q
    - The initial step size ||x^(1) - x^(0)||

    Parameters:
    -----------
    n_iterations : int
        Number of iterations n
    contraction_constant : float
        Contraction constant q (must be 0 < q < 1 for convergence)
    initial_step_norm : float
        Norm of the initial step ||x^(1) - x^(0)||

    Returns:
    --------
    float
        Upper bound on the error ||x^(n) - x̄||

    Raises:
    -------
    ValueError
        If contraction_constant is not in (0, 1)

    Examples:
    ---------
    >>> q = 0.5
    >>> initial_step = 1.0
    >>> n = 10
    >>> error_bound = a_priori_error_bound(n, q, initial_step)
    >>> print(f"After {n} iterations, error <= {error_bound:.6e}")
    """
    if contraction_constant <= 0 or contraction_constant >= 1:
        raise ValueError(f"Contraction constant must be in (0, 1), got {contraction_constant}")

    if n_iterations < 0:
        raise ValueError(f"Number of iterations must be non-negative, got {n_iterations}")

    # Apply a-priori formula
    error_bound = (contraction_constant ** n_iterations / (1 - contraction_constant)) * initial_step_norm

    return error_bound


def a_priori_iterations_needed(tolerance, contraction_constant, initial_step_norm):
    """
    Compute how many iterations are needed to reach a desired tolerance using a-priori estimation.

    Solves: (q^n / (1 - q)) * ||x^(1) - x^(0)|| <= tolerance

    Parameters:
    -----------
    tolerance : float
        Desired error tolerance
    contraction_constant : float
        Contraction constant q (must be 0 < q < 1)
    initial_step_norm : float
        Norm of the initial step ||x^(1) - x^(0)||

    Returns:
    --------
    int
        Minimum number of iterations needed to guarantee error <= tolerance

    Examples:
    ---------
    >>> q = 0.5
    >>> initial_step = 1.0
    >>> tol = 1e-6
    >>> n = a_priori_iterations_needed(tol, q, initial_step)
    >>> print(f"Need {n} iterations to reach tolerance {tol}")
    """
    if contraction_constant <= 0 or contraction_constant >= 1:
        raise ValueError(f"Contraction constant must be in (0, 1), got {contraction_constant}")

    if tolerance <= 0:
        raise ValueError(f"Tolerance must be positive, got {tolerance}")

    # Solve: q^n <= tolerance * (1 - q) / ||x^(1) - x^(0)||
    # n >= log(tolerance * (1 - q) / ||x^(1) - x^(0)||) / log(q)

    rhs = tolerance * (1 - contraction_constant) / initial_step_norm

    if rhs <= 0:
        return float('inf')  # Cannot reach tolerance

    n = np.log(rhs) / np.log(contraction_constant)

    # Round up to get minimum integer iterations
    return int(np.ceil(n))


def compute_initial_step(x0, iteration_function, *args, **kwargs):
    """
    Helper function to compute ||x^(1) - x^(0)|| for a-priori estimation.

    Parameters:
    -----------
    x0 : array_like
        Initial guess
    iteration_function : callable
        Function that computes x^(1) = F(x^(0))
        Should have signature: iteration_function(x, *args, **kwargs)
    norm_type : norm type, optional
        Type of norm to use (default: np.inf)

    Returns:
    --------
    tuple
        (x1, norm) where x1 is the first iteration and norm is ||x^(1) - x^(0)||
    """
    norm_type = kwargs.pop('norm_type', np.inf)

    x0 = np.array(x0, dtype=float)
    x1 = iteration_function(x0, *args, **kwargs)
    x1 = np.array(x1, dtype=float)

    initial_step_norm = np.linalg.norm(x1 - x0, ord=norm_type)

    return x1, initial_step_norm


if __name__ == "__main__":
    print("=" * 70)
    print("A-PRIORI ERROR ESTIMATION - EXAMPLE")
    print("=" * 70)

    # Example: Iterative method with known contraction constant
    q = 0.3  # Contraction constant
    initial_step = 3.5  # ||x^(1) - x^(0)||

    print(f"\nContraction constant q = {q}")
    print(f"Initial step ||x^(1) - x^(0)|| = {initial_step}")

    # Compute error bounds for different numbers of iterations
    print("\n" + "-" * 70)
    print(f"{'n':<5} {'q^n':<15} {'Error Bound':<20}")
    print("-" * 70)

    for n in [0, 1, 2, 5, 10, 20, 50]:
        error_bound = a_priori_error_bound(n, q, initial_step)
        q_power = q ** n
        print(f"{n:<5} {q_power:<15.6e} {error_bound:<20.6e}")

    # Compute iterations needed for specific tolerance
    print("\n" + "=" * 70)
    print("ITERATIONS NEEDED FOR TOLERANCE")
    print("=" * 70)

    for tol in [1e-3, 1e-6, 1e-9, 1e-12]:
        n_needed = a_priori_iterations_needed(tol, q, initial_step)
        actual_error = a_priori_error_bound(n_needed, q, initial_step)
        print(f"\nTolerance: {tol:.0e}")
        print(f"  Iterations needed: {n_needed}")
        print(f"  Actual error bound: {actual_error:.6e}")
        print(f"  Satisfies tolerance: {actual_error <= tol}")

    # Comparison: Show exponential convergence
    print("\n" + "=" * 70)
    print("EXPONENTIAL CONVERGENCE DEMONSTRATION")
    print("=" * 70)
    print("\nNotice how the error decreases exponentially with n:")
    print("Each iteration multiplies the error by approximately q")

    prev_bound = None
    for n in range(1, 6):
        bound = a_priori_error_bound(n, q, initial_step)
        if prev_bound is not None:
            ratio = bound / prev_bound
            print(f"n={n}: error ≤ {bound:.6e}, ratio to previous: {ratio:.4f} ≈ {q}")
        else:
            print(f"n={n}: error ≤ {bound:.6e}")
        prev_bound = bound
