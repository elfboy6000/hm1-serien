import matplotlib.pyplot as plt
import Yanik_Michael_S01_Aufg2 as aufg2

plt.figure(figsize=(10, 5))
# x = interval
# p = polynom values
# dp = derivative values
# pint = integral values

[x, p, dp, pint] = aufg2.yanik_michael_s01_aufg1([2, 1, 3], -10, 10)

# create description, diagram and graph of functions
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('polynomial function')
plt.plot(x, p, label="f(x)") 
plt.plot(x, dp, label="f'(x)")
plt.plot(x, pint, label="F(x)")

plt.xlim(-10, 10)    
plt.ylim(-200,200)
plt.grid(True)
plt.legend()
plt.show()
