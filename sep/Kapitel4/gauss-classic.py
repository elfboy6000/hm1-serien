"""
Classic Gaussian Elimination Algorithm
Step-by-step elimination of each row, then solving for x
"""
import numpy as np

def print_matrix(matrix, title=""):
    """Print matrix in readable format"""
    if title:
        print(f"\n{title}")
    for row in matrix:
        print([f"{val:8.2f}" if abs(val) > 1e-10 else f"{'0':>8}" for val in row])

def gaussian_elimination_classic(A, b):
    #Returns: x: Solution vector

    # Convert to float arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Create augmented matrix [A|b]
    augmented = np.column_stack([A, b])

    print("\n--- INITIAL AUGMENTED MATRIX [A|b] ---")
    print_matrix(augmented)

    # FORWARD ELIMINATION - Eliminate row by row
    for col in range(n - 1):
        # Get pivot element
        pivot = augmented[col, col]

        if abs(pivot) < 1e-10:
            raise ValueError(f"Zero pivot at position ({col}, {col}). Cannot continue.")

        # Eliminate all rows below this column
        for row in range(col + 1, n):

            # Calculate multiplier
            multiplier = augmented[row, col] / pivot

            # Perform elimination: R_row = R_row - multiplier * R_col
            augmented[row] = augmented[row] - multiplier * augmented[col]

    # Show matrix state after elimination
    print("\n--- ELIMINATED AUGMENTED MATRIX [A|b] ---")
    print_matrix(augmented)

    # Solve Det
    determinant = det(augmented)
    print(f"\nDeterminant of A: {determinant:.2f}")

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        # Start with b value
        x[i] = augmented[i, n]

        # Subtract known x values
        if i < n - 1:
            for j in range(i + 1, n):
                x[i] -= augmented[i, j] * x[j]

        # Divide by coefficient
        coeff = augmented[i, i]
        x[i] = x[i] / coeff

    A_original = augmented[:, :-1]
    b_original = np.array(b)

    result = A_original @ x

    print(f"\nSolution x = {x}")

    return x

def det(A_upper_triangle) -> float:
    determinant = 1
    for i in range(0, len(A_upper_triangle)):
        determinant *= A_upper_triangle[i][i]
    return determinant


if __name__ == "__main__":
    A = [
        [-1, 1, 1],
        [1, -3, -2],
        [5, 1, 4]
    ]

    b = [0, 5, 3]

    x = gaussian_elimination_classic(A, b)