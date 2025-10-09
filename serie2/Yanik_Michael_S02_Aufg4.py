# machine epsilon: largest tiny number with 1+eps == 1
eps = 1.0
while 1.0 + eps/2.0 != 1.0:
    eps /= 2.0

# q_min: smallest huge number with 1+q == q
q = 1.0
while q + 1.0 != q:
    q *= 2.0
low, high = q/2.0, q
for _ in range(80):  # binary search
    m = (low + high) / 2.0
    if m + 1.0 == m: high = m
    else: low = m
q_min = high

print("eps   =", eps)
print("q_min =", q_min)
print("eps*q_min ≈", eps * q_min)
