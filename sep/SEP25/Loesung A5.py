# -*- coding: utf-8 -*-
"""
@author: roor
"""

import sys

import numpy as np

np.set_printoptions(precision=4)

def sor(A, b, x0, omega, tol):

    def norm(x):
        return np.linalg.norm(x, 1)

    D = np.diag(np.diag(A))
    R = np.triu(A, k=1)
    L = np.tril(A, k=-1)

    M1_inv = np.linalg.inv(1/omega*D)
    M2 = -((omega-1)/omega*D + L + R)

    B = M1_inv @ M2
    c = M1_inv @ b

    print('B=\n', B)
    print('c=\n', c)

    B_norm = norm(B)

    # Test, ob das Verfahren konvergiert; falls nicht, Abbruch mit error()
    if B_norm > 1:
        print('x ist kein anziehender Fixpunkt')
        sys.exit(1)

    xn = B @ x0 + c
    n = 1
    n2 = np.ceil(np.log(tol*(1-B_norm)/norm(xn-x0)) / np.log(B_norm))
    # print(f'{n}: Inf-norm = {B_norm / (1 - B_norm) * norm(xn - x0)}')

    while B_norm/(1-B_norm)*norm(xn-x0) >= tol:
        x0 = xn
        xn = B @ x0 + c
        n = n+1
        # print(f'{n}: Inf-norm = {B_norm/(1-B_norm)*norm(xn-x0)}')

    return xn, n, n2, B_norm


if __name__ == '__main__':

    # A = np.array([[5,1,-1],[1,7,1],[-1,1,5]])
    diag_element = 7
    A = np.array([[diag_element,-2,-2],[-2,diag_element,-2],[-2,-2,diag_element]])
    x_sol = np.array([1, -1, 2])
    b = np.array([5, -13, 14])
    tol = 1.0e-9

    x0 = np.array([0, 0, 0])
    print(f'\nLösung für Standard-Jacobi-Verfahren:\n')
    x_sol, n_standard, _, B_norm = sor(A, b, x0, 1.0, tol)
    print(f'\nn={n_standard}, s_sol={x_sol}, B_norm={B_norm:.4f}\n')

    omega = 1.15
    x0 = np.array([0, 0, 0])
    print(f'\nLösung für alternative Methode mit omega={omega}:\n')
    x_sol, n_alternativ, _, B_norm = sor(A, b, x0, omega, tol)
    print(f'\nn={n_alternativ}, s_sol={x_sol}, B_norm={B_norm:.4f}\n')

    if n_standard>n_alternativ:
        print(f'Lösung mit Alternative braucht nur {n_alternativ} statt {n_standard} Iterationen\n')
    else:
        print(f'Keine Verbesserung')

    print('end')


"""
Musterlösung:
=============

Lösung für Standard-Jacobi-Verfahren:
B=                                                      (1 P)
 [[0.     0.2857 0.2857]
 [0.2857 0.     0.2857]
 [0.2857 0.2857 0.    ]]
 
c=                                                      (1 P)
 [ 0.7143 -1.8571  2.    ]
 
Korrekte Implementierung Abbruchbedingung               (1 P)

n=39, s_sol=[ 1. -1.  2.], B_norm=0.5714                (2 P)



Korrekte Implementierung Alternatives Verfahren         (1 P)

Lösung für alternative Methode mit omega=1.15:
B=                                                      (1 P)
 [[-0.15    0.3286  0.3286]
 [ 0.3286 -0.15    0.3286]
 [ 0.3286  0.3286 -0.15  ]]
 
c=                                                      (1 P)
 [ 0.8214 -2.1357  2.3   ]
 
n=34, s_sol=[ 1. -1.  2.], B_norm=0.8071                (2 P)

Lösung mit Alternative braucht nur 34 statt 39 Iterationen

"""