import numpy as np
import matplotlib.pyplot as plt

# h(x) given
def h_naive(x):
    return np.sqrt(100.0*x*x - 200.0*x + 99.0)

# Exact algebraic rewrite to avoid cancellation near x≈1.1:
def h_stable(x):
    return np.sqrt(100.0*(x-1.0)**2 - 1.0)

# (a)
x_a = np.linspace(1.0999, 1.1001, 801)
yN = h_naive(x_a)
yS = h_stable(x_a)
rel_err = np.abs(yN - yS)/np.maximum(np.abs(yS), 1e-300)

plt.figure(); plt.plot(x_a, yN, label="naive"); plt.plot(x_a, yS, "--", label="stable")
plt.title("h(x) near 1.1 — cancellation vs. stable form")
plt.xlabel("x"); plt.ylabel("h(x)"); plt.legend(); plt.grid(True); plt.show()

plt.figure(); plt.semilogy(x_a, rel_err)
plt.title("Relative error |h_naive - h_stable| / |h_stable|"); plt.xlabel("x"); plt.grid(True); plt.show()

# (b)
# From h(x)=sqrt(100(x-1)^2-1):
# h'(x) = 100(x-1)/h(x)
# k(x) = | 100 x (x-1) / (100(x-1)^2 - 1) |
dx = 1e-7
x_b = np.arange(1.1, 1.3 + dx/2, dx, dtype=np.float64)
den = 100.0*(x_b - 1.0)**2 - 1.0
kappa = np.abs(100.0 * x_b * (x_b - 1.0) / den)

plt.figure(); plt.semilogy(x_b, kappa)
plt.title(r"Condition $\kappa(x)=|100\,x(x-1)/(100(x-1)^2-1)|$")
plt.xlabel("x"); plt.ylabel("kappa(x)"); plt.grid(True, which="both"); plt.show()

# (c)
# The algebraic rewrite removes cancellation (algorithmic error),
# but k(x)->Infinity as x->1.1 (denominator->0): the problem is ill-conditioned there,
# so sensitivity cannot be eliminated—only reduced by using the stable form.
