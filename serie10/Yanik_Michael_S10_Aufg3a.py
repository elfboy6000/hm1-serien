import numpy as np

def norm(A):
    return np.linalg.norm(A, np.inf)

def Yanik_Michael_S10_Aufg3a(A, b, x0, tol, opt):
    # A-Priori
    [B, C] = calculate_BC(A, b, opt)
    x1 = iteration(B, C, x0)
    n2 = a_priori(B, x1, x0, tol)

    xn = x1
    x_before = x0
    n = 1
    while norm(xn - x_before) > tol and n < n2:
        n += 1
        x_before = xn
        xn = iteration(B, C, xn)
        print(xn)

    return xn, n, n2


def a_priori(B, x1, x0, tol):
    B_norm = norm(B)
    x_norm = norm(x1-x0)
    return np.ceil(np.log(-(B_norm-1)*tol/x_norm)/np.log(B_norm))

def calculate_BC(A,b, opt):
    [L, D, R] = split(A)
    # GaussSeidel
    if (opt):
        DL_inv = np.linalg.inv(D+L)
        return [np.dot(-DL_inv,R), np.dot(DL_inv,b)]
    # Jacobi
    else:
        D_inv = np.linalg.inv(D)
        return [np.dot(-D_inv,(L+R)), np.dot(D_inv,b)]


def iteration(B,C,x_last):
    return np.dot(B, x_last) + C

def split(A):
    size = A.shape[0]
    L = np.zeros((size, size))
    D = np.zeros((size, size))
    R = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i == j):
                D[i][j] = A[i][j]
            elif (i > j):
                L[i][j] = A[i][j]
            else:
                R[i][j] = A[i][j]
    return L, D, R

if __name__ == "__main__":
    A = np.array([
        [8,5,2],
        [5,9,1],
        [4,2,7],
    ])
    b = np.array([19,5,34])
    x0 = np.array([1,-1,3])
    tol = 10**-4

    [xn, n, n2] = Yanik_Michael_S10_Aufg3a(A, b, x0, tol, 1)
    print(xn)
    print(n)
    print(n2)