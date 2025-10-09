import numpy as np
import matplotlib.pyplot as plt

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
plt.title('g(x) = x / (sin(1+x) - sin(1))')
plt.title('g(x) = x / (2*cos(1+x/2)*sin(x/2))')
plt.plot(range_alt, g_transformed(range_alt), label='f(x)')
plt.plot(range_alt, g_direct(range_alt), label='f(x)')
plt.grid(True)
plt.legend()
plt.show()