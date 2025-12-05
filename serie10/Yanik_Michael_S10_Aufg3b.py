import numpy as np
import matplotlib.pyplot as plt
import Yanik_Michael_S10_Aufg3a as s10
import serie6.Yanik_Michael_S06_Aufg2 as s6

dim = 300
A = np.diag(np.diag(np.ones((dim,dim))*4000))+np.ones((dim,dim))
dum1 = np.arange(1,np.int_(dim/2+1),dtype=np.float64).reshape((np.int_(dim/2),1))
dum2 = np.arange(np.int_(dim/2),0,-1,dtype=np.float64).reshape((np.int_(dim/2),1))
x = np.append(dum1,dum2,axis=0)
b = A@x
x0 = np.zeros((dim,1))
tol = 1e-4

print("Begin Timing S10")
import timeit
t1=timeit.timeit("s10.Yanik_Michael_S10_Aufg3a(A, b, x0, tol, 1)", "from __main__ import s10, A, b, x0, tol", number=1)
print("Begin Timing S6")
t2=timeit.timeit("s6.Yanik_Michael_S06_Aufg2(A, b)", "from __main__ import s6, A, b", number=1)
print("Begin Timing Numpy")
t3=timeit.timeit("np.linalg.solve(A, b)", "from __main__ import np, A, b", number=1)
print("End Timing. Results:")
print(t1)
print(t2)
print(t3)
# End Timing. Results:
# 0.035603290976723656
# 0.9857576669892296
# 0.0029622090223710984
# Zerlegung braucht ungefähr 28x länger als Gauss-Seidel

[x1, n, n2] = s10.Yanik_Michael_S10_Aufg3a(A, b, x0, tol, 1)
[x2, n, n2] = s10.Yanik_Michael_S10_Aufg3a(A, b, x0, tol, 0)
[A, det, x3] = s6.Yanik_Michael_S06_Aufg2(A, b)
x3 = np.array([x3]).reshape((dim, 1))

y1 = np.abs(x - x1)
y2 = np.abs(x - x2)
y3 = np.abs(x - x3)
x_values = np.arange(dim)
plt.plot(x_values, y1)
plt.plot(x_values, y2)
plt.plot(x_values, y3)
plt.legend(["Gauss-Seidel", "Jacobi", "Gauss"])
plt.show()
