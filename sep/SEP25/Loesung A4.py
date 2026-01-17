# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:08:43 2024

@author: delo
"""
#=====================
# Version A (delo)
#=====================


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
 

#%% a)
x = sp.symbols('x')
F_sym = (sp.exp(x) + sp.exp(-x))/2 - 3/2

F = sp.lambdify(x, F_sym, 'numpy')

tol = 1e-6
n = 0
x0 = 1.5
while True:
    x1 = F(x0)
    n += 1
    print(x1, n)

    if abs(x1-x0) < tol:            # 1P
        break
    x0 = x1

print("x_{}={:.6f}".format(n,x1))
# x_18=-0.413348                     1P

#%%
# Plot
dF_sym = F_sym.diff(x)                # 1P
dF = sp.lambdify(x, dF_sym, 'numpy')


a = -1
b = 2.5
x = np.linspace(a,b)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, F(x), 'r', x, x, 'k')
ax1.legend(('y=F(x)','y=x'))              
ax2.plot(x, np.abs(dF(x)), 'r', x, x**0, 'k')  # 1P
ax2.legend(("y=|F'(x)|",'y=1'))
ax2.set_xlabel('x')
# Der nahe FP1 in [1.5,2] ist abstossend, |F'| > 1
# Iteration wird von FP1 weggedrückt (Richtung anziehenden FP2 in [-1,0])     1P

# %% b)

f = lambda x : F(x) - x      # 2P
x0 = 1.4
x1 = 1.6
n = 1
while abs(f(x1)) > tol:             # 1P
    n += 1
    x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
    x1, x0 = x2, x1
    print(x1, n)

print("x_{}={:.6f}".format(n,x1))
# x_7=1.892136                      1P
