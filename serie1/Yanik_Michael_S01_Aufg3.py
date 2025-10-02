import timeit

def fact_rec(n):
# y = fact_rec(n) berechnet die Fakultät von n als fact_rec(n) = n * fact_rec(n -1) mit fact_rec(0) = 1
# Fehler, falls n < 0 oder nicht ganzzahlig
    import numpy as np
    if n < 0 or np.trunc(n) != n:
        raise Exception('The factorial is defined only for positive integers')
    if n <=1:
        return 1
    else:
        return n*fact_rec(n-1)

def fact_for(n):
# y = fact_for(n) berechnet die Fakultät von n als fact_for(n) = n * (n-1) * ... * 1 mit fact_for(0) = 1
# Fehler, falls n < 0 oder nicht ganzzahlig
    import numpy as np
    n = int(n)
    if n < 0 or np.trunc(n) != n:
        raise Exception('The factorial is defined only for positive integers')
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

t1=timeit.repeat("fact_rec(500)", "from __main__ import fact_rec", number=100)
t2=timeit.repeat("fact_for(500)", "from __main__ import fact_for", number=100)
print("fact_rec: min time for 100 runs of fact_rec(500): ", min(t1))
print("fact_for: min time for 100 runs of fact_for(500): ", min(t2))

# 1) Welche der beiden Funktionen ist schneller und um was für einen Faktor? Weshalb?
# fact_for is faster than fact_rec because function calls are more expensive than loops in Python

# 2) Gibt es in Python eine obere Grenze für die Fakultät von n
#  a) als ganze Zahl (vom Typ 'integer')? Versuchen Sie hierzu, das Resultat für n ∈ [190, 200] als integer auszugeben.
for n in range(190, 201):
    try:
        print(f"fact_for({n}) = {fact_for(n)}")
    except OverflowError as e:
        print(f"fact_for({n}) caused an overflow error: {e}")
    try:
        print(f"fact_rec({n}) = {fact_rec(n)}")
    except RecursionError as e:
        print(f"fact_rec({n}) caused a recursion error: {e}")
# There is no limit using integers within this range

#  b) als reelle Zahl (vom Typ 'float')? Versuchen Sie hierzu, das Resultat für n ∈ [170, 171] als float auszugeben.
for n in range(170, 172):
    try:
        print(f"fact_for({n}) = {float(fact_for(n))}")
    except OverflowError as e:
        print(f"fact_for({n}) caused an overflow error: {e}")
    try:
        print(f"fact_rec({n}) = {float(fact_rec(n))}")
    except RecursionError as e:
        print(f"fact_rec({n}) caused a recursion error: {e}")
# "fact_for(171) caused an overflow error: int too large to convert to float"