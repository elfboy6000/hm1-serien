# -*- coding: utf-8 -*-
"""
Loesung Aufgabe 6, Version A

@author: beer
"""


"""
Aufgabe a):
-----------
"""

import numpy as np

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)

# Parameter c = -3
A = np.array([[ 30., -3],
              [-13,   4]])

# Startvektor
v = np.array([[1], [0]])
v_prev = np.array([[0], [0]])

n = 0
while np.linalg.norm(v - v_prev) >= 1e-15:
    v_prev = v
    
    v = A@v
    
    lamb = (v_prev.T@v)/(v_prev.T@v_prev)
    
    v = v/np.linalg.norm(v, 2)
    n += 1

"""
Bewertung Script:                                                               (2 P)
"""
    
print("Eigenwert lambda: ", lamb[0,0])  # 31.422205 = 17+4*sqrt(17)             (1 P)
print("Eigenvektor v: ")
print(v)                                # v = [[0.9036035], [-0.42836984]]      (1 P)
print("Anzahl Iterationen: ", n)        # n = 15                                (1 P)




"""
Aufgabe b):
-----------
"""

"""

A = [[ 30, c],
     [-13, 4]]


Das charakteristische Polynom lautet: p(x) = x^2 - 34x + 120+13c.               (1 P)

Eine Doppelloesung und damit einen Eigenwert mit algebr. Vielfachheit 2
erhaelt man genau dann, wenn die Diskriminante D = 34^2 - 4(120 + 13c) = 0.
Damit erhaelt man c = 13 und den Eigenwert lambda = 17.                         (2 P)

Die Zeilenstufenform der Matrix A = [[ 30-17, 13],
                                     [-13,     4-17]] lautet: 
    [[ 13, 13],
     [  0,  0]].
Daraus ist ersichtlich, dass die geometrische Vielfachheit nur 1 betraegt,
und somit A nicht diagonalisierbar ist.                                         (2 P)
"""
