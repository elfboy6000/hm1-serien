import numpy as np

# a)
A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
B = np.array([[5, 4, 3, 2], [4, 3, 2, 5], [3, 2, 5, 4], [2, 5, 4, 3]])
b = np.array([1, 2, 3, 4]).reshape(4, 1)

print("\nAb:\n", A @ b)
print("\nBb:\n", B @ b)
print("\nA^T:\n", A.T)
print("\nB^T:\n", B.T)
print("\nA^TA:\n", A.T @ A)
print("\nB^TB:\n", B.T @ B)

# b)
result = A[3, :] @ B[:, 1]
print("\nb)", result)

# c)

sum_A = np.sum(A, axis=0)
sum_B = np.sum(B, axis=1)
print("\nc)\n Summe Spalten A:\n", sum_A)
print("Summe Zeilen B:\n", sum_B)