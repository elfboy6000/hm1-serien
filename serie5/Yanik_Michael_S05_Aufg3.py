import numpy as np

def Yanik_Michael_S05_Aufg3(f, x0, x1, tol):
    a, b = x0, x1
    fa, fb = f(a), f(b)
    while abs(fb) > tol:
        if fb == fa:
            break
        c = b - fb * (b - a) / (fb - fa)
        a, fa, b, fb = b, fb, c, f(c)
    return b

# Aufgabe 1
def equation1(x): return np.exp(x**2) + x**-3 - 10
r1 = Yanik_Michael_S05_Aufg3(equation1, -1.0, -1.2, 1e-6)
print("Zero from Aufgabe 1:", r1)

# Aufgabe 2
def equation2(x): return -(1/3)*np.pi*x**3 + 5*np.pi*x**2 - 471
r2 = Yanik_Michael_S05_Aufg3(equation2, 5, 7, 1e-6)
print("Zero from Aufgabe 2:", r2)

"""
Newton’s method issues:
- Needs derivative f'(x) (often unavailable or costly)
- Fails if f'(x) ≈ 0 → large/diverging steps
- Can diverge or jump outside domain (e.g., near x=0 in Aufgabe 1)
Secant avoids these by approximating f'(x) numerically.
"""
