import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


range_1 = np.linspace(1.99, 2.01, 501)

def explicit_function(x):
    return x**7 - 14*x**6 + 84*x**5 - 280*x**4 + 560*x**3 - 672*x**2 + 448*x - 128

# coeffs_1 = [1, -14, 84, -280, 560, -672, 448, -128]
coeffs_2 = np.poly1d([1, -2])
# f1_1 = np.poly1d(coeffs_1)
f1_2 = reduce(np.polymul, [coeffs_2]*7) 

plt.figure(1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x) = (x-2)^7')
plt.title('f(x) = x^7 - 14x^6 + 84x^5 - 280x^4 + 560x^3 + 672x^2 + 448x - 128')
plt.plot(range_1, explicit_function(range_1), label='f(x) - (explicit function)')
plt.plot(range_1, f1_2(range_1), label='f(x) - (x-2)^7')
# plt.plot(range_1, prime_f(range_1), label="f'(x) - derivative")
# plt.plot(range_1, deriv_f(range_1), label='F(x) - integral')
plt.grid(True)
plt.legend()
plt.show()

# Antwort zu a)
# Aufgrund von Rundungsfehlern in der numerischen Berechnung weichen die beiden Polynome voneinander ab.
# Die erste Funktion beinhaltet viele Terme mit grossen Koeffizienten, die sich bei kleinen x-Werten stark gegenseitig aufheben.
# Die zweite Funktion setzt die x-Werte direkt in die Klammer (x-2) ein, was numerisch stabiler ist.

def g_direct(x):
    return x / (np.sin(1.0 + x) - np.sin(1.0))

def g_transformed(x):
    return x / (2.0 * np.cos(1.0 + x / 2.0) * np.sin(x / 2.0))

start = -1e-14
stop  =  1e-14
step  =  1e-17

start_alt = -10**(-14)
stop_alt = 10**(-14)
step_alt = 10**(-17)

range = np.arange(start, stop, step)
range_alt = np.arange(start_alt, stop_alt, step_alt)

plt.figure(1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x) = x / (sin(1+x) - sin(1))')
plt.title('g(x) = x / (2*cos(1+x/2)*sin(x/2))')
plt.plot(range, g_direct(range), label='f(x)')
plt.plot(range, g_transformed(range), label='g(x)')
#plt.plot(range_alt, g_direct(range_alt), label='f(x)')
#plt.plot(range_alt, g_transformed(range_alt), label='g(x)')
plt.grid(True)
plt.legend()
plt.show()

# Antwort zu b) und c)
# Die erste Funktion g_directergibt bei x gegen 0 eine 0/0 Situation und ist somit numerisch instabil.
# Die zweite Funktion g_transformed ist stabiler, da sie den Ausdruck umformt und somit die 0/0 Situation vermeidet.
