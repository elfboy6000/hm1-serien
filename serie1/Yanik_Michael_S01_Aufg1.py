import numpy as np
import matplotlib.pyplot as plt

# Exercise 1

plt.figure(1)
# create f(x) = x^5 - 5x^4 - 30x^3 + 110x^2 + 29x - 105
range_1 = np.arange(-10,10,0.1)
coeffs_1 = [1, -5, -30, 110, 29, -105]
f1_1 = np.poly1d(coeffs_1)
# create  f'(x) and F(x) with integration constant = 0
prime_f = f1_1.deriv()
deriv_f = f1_1.integ()  # Integration constant = 0 by default
# create description, diagram and graph of functions
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Function f(x) = x^5 - 5x^4 - 30x^3 + 110x^2 + 29x - 105')
plt.plot(range_1, f1_1(range_1), label='f(x) - polynomial') 
plt.plot(range_1, prime_f(range_1), label="f'(x) - derivative")
plt.plot(range_1, deriv_f(range_1), label='F(x) - integral')

plt.xlim(-10,10)
plt.ylim(-200,200)
plt.grid(True)

plt.legend()
plt.show()

