import math
import matplotlib.pyplot as plt

# a)
def side_calculator(prev_side_length):
    return math.sqrt(2 - 2 * math.sqrt(1 - (prev_side_length / 2) ** 2))

sides = 6;
prev_side_length = 1

plt.figure(figsize=(10, 10))
plt.title("Side length of inscribed polygon vs number of sides")
plt.xlabel("Number of sides")
plt.ylabel("Side length")
x = [sides * 2**i for i in range(0, 20)]
y = []
current = prev_side_length
for i in range(0, 20):
    y.append(current)
    current = side_calculator(current)
plt.plot(x, y, marker='o')
plt.xscale('log', base=2)
plt.yscale('linear')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Was passiert für grosse n und weshalb?
# The side length approaches 0 as the number of sides increases, because the inscribed polygon approaches the shape of the circle.

# b)
def new_side_calculator(prev_side_length):
    return math.sqrt(prev_side_length ** 2 / 2 * (1 + math.sqrt(1 - (prev_side_length / 2) ** 2)))

plt.figure(figsize=(10, 10))
plt.title("New Side length of inscribed polygon vs number of sides")
plt.xlabel("Number of sides")
plt.ylabel("Side length")
x = [sides * 2**i for i in range(0, 20)]
y = []
current = prev_side_length
for i in range(0, 20):
    y.append(current)
    current = new_side_calculator(current)
plt.plot(x, y, marker='o')
plt.xscale('log', base=2)
plt.yscale('linear')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Was beobachten Sie?
# The side length approaches 0 much slower and more linearly