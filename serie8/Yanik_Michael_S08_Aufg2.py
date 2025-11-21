# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:26:09 2020

Höhere Mathematik 1, Serie 8, Gerüst für Aufgabe 2

Description: calculates the QR factorization of A such that A = QR
Input Parameters: A: array, n*n matrix
Output Parameters: Q : n*n orthogonal matrix
                   R : n*n upper right triangular matrix            
Remarks: none
Example: A = np.array([[1,2,-1], [4,-2,6], [3,1,0]]) 
        [Q,R] = Serie08_Aufg2(A)

@author: Yanik_Michael
"""

import numpy as np
import timeit


def Serie08_Aufg2(A):

    A = np.copy(A)                              #necessary to prevent changes to original matrix A
    A = A.astype('float64')                     #enforce data type float

    n = np.shape(A)[0]

    if n != np.shape(A)[1]:
        raise Exception('Matrix is not square')

    Q = np.eye(n)
    R = A

    for j in range(0, n-1):
        a = np.copy(R[j:, j]).reshape(n - j, 1)        #reshape to column vectors
        e = np.zeros((n - j, 1))
        e[0, 0] = 1.0
        length_a = np.linalg.norm(a)
        if a[0, 0] >= 0:
            sig = 1.0
        else:
            sig = -1.0
        v = a + sig * length_a * e
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        u = v / v_norm
        H = np.eye(n - j) - 2.0 * (u @ u.T)
        Qi = np.eye(n)
        Qi[j:, j:] = H
        R = Qi @ R
        Q = Q @ Qi.T

    return(Q,R)


def IT24a_WIN_11_S08_Aufg2(A):
    return Serie08_Aufg2(A)


def solve_Aufgabe1():
    A = np.array([[1, -2, 3],
                  [-5,  4, 1],
                  [2, -1, 3]], dtype=float)
    b = np.array([1, 9, 5], dtype=float)

    Q, R = Serie08_Aufg2(A)

    print("Q =")
    print(Q)
    print("\nR =")
    print(R)
    print("\nCheck A = Q @ R:")
    print(Q @ R)

    # Rx = Q^T b
    y = Q.T @ b
    n = len(b)
    x = np.zeros(n)

    # reverse insertion
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    print("\nSolution x of Ax = b:")
    print(x)

def duration_comparison():
    # c) Laufzeit für A (3x3)
    t1 = timeit.repeat("IT24a_WIN_11_S08_Aufg2(A)",
                       "from __main__ import IT24a_WIN_11_S08_Aufg2, A",
                       number=100)
    t2 = timeit.repeat("np.linalg.qr(A)",
                       "from __main__ import np, A",
                       number=100)

    avg_t1 = np.average(t1) / 100.0
    avg_t2 = np.average(t2) / 100.0

    print("\nAverage time per call for A (3x3):")
    print(f"own QR function:   {avg_t1:.6e} s")
    print(f"numpy.linalg.qr(A):   {avg_t2:.6e} s")

    # d) Runtime comparison for artificial 100x100 matrix
    t1_big = timeit.repeat("IT24a_WIN_11_S08_Aufg2(Test)",
                           "from __main__ import IT24a_WIN_11_S08_Aufg2, Test",
                           number=100)
    t2_big = timeit.repeat("np.linalg.qr(Test)",
                           "from __main__ import np, Test",
                           number=100)

    avg_t1_big = np.average(t1_big) / 100.0
    avg_t2_big = np.average(t2_big) / 100.0

    print("\nAverage time per call for test (100x100):")
    print(f"Own QR function:   {avg_t1_big:.6e} s")
    print(f"numpy.linalg.qr(Test):{avg_t2_big:.6e} s")

if __name__ == "__main__":
    # Define matrices globally for timeit
    A = np.array([[1, -2, 3],
                  [-5, 4, 1],
                  [2, -1, 3]], dtype=float)
    Test = np.random.rand(100, 100)

    # Check the solution of Aufgabe 1
    solve_Aufgabe1()

    # duration comparison for c) and d)
    duration_comparison()

"""
        For a 100x100 matrix (test = np.random.rand(100,100)), numpy.linalg.qr() is significantly faster because it is 
        implemented in optimised C/Fortran (BLAS/LAPACK), whereas this function works in pure Python and explicitly 
        performs each Householder reflection.
    """
