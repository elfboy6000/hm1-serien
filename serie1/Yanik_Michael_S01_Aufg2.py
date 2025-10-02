import numpy as np
import matplotlib.pyplot as plt

def yanik_michael_s01_aufg1(a, xmin, xmax):

    # coeff array
    a = np.array(a, dtype=float)

    def horner(coeffs, x):
        result = 0
        for c in coeffs:
            result = result * x + c
        return result

    def deriv(coeffs):
        if len(coeffs) <= 1:
            return np.array([0])
        return np.array([coeffs[i] * (len(coeffs) - 1 - i) for i in range(len(coeffs) - 1)])

    def integ(coeffs):
        new_coeffs = [coeffs[i] / (len(coeffs) - i) for i in range(len(coeffs))]
        return np.array(new_coeffs + [0]) 

    # interval values
    x = np.arange(xmin, xmax + 0.01, 0.01) 

    # polynom
    p = np.array([horner(a, i)for i in x])
    # derivative
    dp = np.array([horner(deriv(a), i)for i in x])
    # integral
    pint = np.array([horner(integ(a), i)for i in x])

    return (x, p, dp, pint)
