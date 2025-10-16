import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 5 / np.cbrt(2*x**2)
g = lambda x: 10**5 * (2*np.e)**(-x/100)
h = lambda x: 16**x

x1 = np.logspace(-2, 2, 400)
plt.figure()
plt.loglog(x1, f(x1))
plt.title("f(x) on log-log → straight line")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.show()

x2 = np.linspace(1e-6, 100, 600)
plt.figure()
plt.semilogy(x2, g(x2))
plt.title("g(x) on semilogy → straight line")
plt.xlabel("x"); plt.ylabel("g(x)")
plt.show()

plt.figure()
plt.semilogy(x2, h(x2))
plt.title("h(x) on semilogy → straight line")
plt.xlabel("x"); plt.ylabel("h(x)")
plt.show()
