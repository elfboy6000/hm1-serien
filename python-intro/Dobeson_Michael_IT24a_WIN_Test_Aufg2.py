import matplotlib.pyplot as plt
import numpy as np

# Abbildung 1
plt.figure(1)
x = np.arange(-5, 5, 0.05)
f = np.exp(x)
plt.plot(x, f)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Exponential Function')
plt.legend(['f(x) = e^x'])
plt.show()

# Abbildung 2
plt.figure(2)
x = np.arange(-10, 10, 0.1)
p = x**5 + 3*x**4 + 3*x**2 + x + 1
plt.plot(x, p)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Polynomial Function')
plt.legend(['p(x) = x^5 + 3x^4 + 3x^2 + x + 1'])
plt.show()

# Abbildung 3
plt.figure(3)
x = np.arange(-2*np.pi, 2*np.pi, 0.01)
g = 1/2 * np.sin(3*x)
h = 1/2 * np.cos(3*x)
plt.plot(x, g, label='g(x) = 1/2 * sin(3x)')
plt.plot(x, h, label='h(x) = 1/2 * cos(3x)')
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine Functions')
plt.legend()
plt.show()
