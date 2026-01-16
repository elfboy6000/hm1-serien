"""
A-Posteriori Error Estimation
Based on Banach Fixed Point Theorem

For iterative method x^(n+1) = F(x^(n)) with contraction constant q:
Error bound: ||x^(n) - x̄|| <= (q / (1 - q)) * ||x^(n) - x^(n-1)||

where:
- x^(n) is the current approximation
- x̄ is the exact solution (fixed point)
- q is the contraction constant (e.g., ||B|| for linear iteration)
"""
import numpy as np


def a_posteriori_error_bound(x_current, x_previous, contraction_constant, norm_type=np.inf):
    """
    Compute a-posteriori error bound for iterative methods.

    Formula: ||x^(n) - x̄|| <= (q / (1 - q)) * ||x^(n) - x^(n-1)||

    Parameters:
    -----------
    x_current : array_like
        Current iteration x^(n)
    x_previous : array_like
        Previous iteration x^(n-1)
    contraction_constant : float
        Contraction constant q (must be 0 < q < 1 for convergence)
    norm_type : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Type of norm to use (default: np.inf for infinity norm)

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
    >>> x_current = np.array([1.5, 2.3, 3.1])
    >>> x_previous = np.array([1.4, 2.2, 3.0])
    >>> q = 0.5
    >>> error_bound = a_posteriori_error_bound(x_current, x_previous, q)
    >>> print(f"Error bound: {error_bound:.6e}")
    """
    if contraction_constant <= 0 or contraction_constant >= 1:
        raise ValueError(f"Contraction constant must be in (0, 1), got {contraction_constant}")

    x_current = np.array(x_current, dtype=float)
    x_previous = np.array(x_previous, dtype=float)

    # Compute ||x^(n) - x^(n-1)||
    difference_norm = np.linalg.norm(x_current - x_previous, ord=norm_type)

    # Apply a-posteriori formula
    error_bound = (contraction_constant / (1 - contraction_constant)) * difference_norm

    return error_bound


def a_posteriori_iteration_bound(n_iterations, contraction_constant, initial_error, norm_type=np.inf):
    """
    Compute how many iterations are needed to reach a desired tolerance using a-posteriori estimation.

    Parameters:
    -----------
    n_iterations : int
        Number of iterations to estimate error for
    contraction_constant : float
        Contraction constant q (must be 0 < q < 1)
    initial_error : float
        Initial error ||x^(1) - x^(0)||
    norm_type : norm type, optional
        Type of norm used (for documentation purposes)

    Returns:
    --------
    float
        Estimated error after n iterations
    """
    if contraction_constant <= 0 or contraction_constant >= 1:
        raise ValueError(f"Contraction constant must be in (0, 1), got {contraction_constant}")

    # Error after n iterations
    return (contraction_constant / (1 - contraction_constant)) * initial_error


if __name__ == "__main__":
    print("=" * 70)
    print("A-POSTERIORI ERROR ESTIMATION - EXAMPLE")
    print("=" * 70)

    # Example: Iterative method with known contraction constant
    q = 0.3  # Contraction constant

    # Simulate some iterations
    x_prev = np.array([0.0, 0.0, 0.0])
    x_curr = np.array([1.25, 2.75, 3.0])

    print(f"\nContraction constant q = {q}")
    print(f"Previous iteration x^(n-1) = {x_prev}")
    print(f"Current iteration x^(n) = {x_curr}")

    # Compute a-posteriori error bound
    error_bound = a_posteriori_error_bound(x_curr, x_prev, q)

    print(f"\n||x^(n) - x^(n-1)||_∞ = {np.linalg.norm(x_curr - x_prev, ord=np.inf):.6f}")
    print(f"A-posteriori error bound: ||x^(n) - x̄||_∞ <= {error_bound:.6f}")
    print(f"\nThis means the true solution x̄ is within {error_bound:.6f} of the current approximation.")

    # Example with different norm types
    print("\n" + "=" * 70)
    print("DIFFERENT NORM TYPES")
    print("=" * 70)

    for norm_type, norm_name in [(1, "1-norm"), (2, "2-norm"), (np.inf, "∞-norm")]:
        bound = a_posteriori_error_bound(x_curr, x_prev, q, norm_type=norm_type)
        print(f"Error bound ({norm_name}): {bound:.6f}")
