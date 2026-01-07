
import numpy as np
from sep.Kapitel4.gauss_classic import gaussian_elimination_classic


def fehlerrechnung(A, b, max_error=1e-10):
    print("A:\n", np.array(A))
    normen_A = np.linalg.norm(A, np.inf)
    print(f"\nNormen von A (∞-Norm): {normen_A:.2f}")

    print("b:\n", np.array(b))
    normen_b = np.linalg.norm(b, np.inf)
    print(f"\nNormen von b (∞-Norm): {normen_b:.2f}")

    inverse_A = np.linalg.inv(A)
    print(f"\nA^-1:\n", inverse_A)
    normen_inverse_A = np.linalg.norm(inverse_A, np.inf)
    print(f"\nNormen von A^-1 (∞-Norm): {normen_inverse_A:.2f}")
    konditionszahl = normen_A * normen_inverse_A
    print(f"\ncond(A) (∞-Norm): {konditionszahl:.2f}")

    fehler = max_error * normen_inverse_A
    print(f"\nx-~x (∞-Norm) <= [A^-1 (∞-Norm) * b-~b (∞-Norm) = {fehler:.2f}]")

    relative_fehler = konditionszahl * (max_error / normen_b)
    print(f"\n(x-~x) / x (∞-Norm) <= cond(A) * ((b-~b) / b (∞-Norm)) = {relative_fehler:.2f}")

if __name__ == "__main__":
    A = [
        [2, 4],
        [4, 8.1]
    ]

    b = [1, 1.5]

    max_error = 0.1

    x = fehlerrechnung(A, b, 0.1)

    incorrect_b = [0.9, 1.6]

    gaussian_elimination_classic(A,b)
    gaussian_elimination_classic(A,incorrect_b)