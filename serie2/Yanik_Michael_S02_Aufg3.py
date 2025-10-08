import math
import matplotlib.pyplot as plt

def side_calculator(prev_side_length):
    return math.sqrt(2 - 2 * math.sqrt(1 - (prev_side_length / 2) ** 2))

def new_side_calculator(prev_side_length):
    return math.sqrt(prev_side_length ** 2 / (2 * (1 + math.sqrt(1 - (prev_side_length / 2) ** 2))))

sides = 6;
prev_side_length = 1.0

plt.figure()
plt.title("Perimeter of inscribed polygon vs number of sides")
plt.xlabel("Number of sides")
plt.ylabel("Perimeter")
x = []
y = []
x2 = []
y2 = []
current = prev_side_length
for i in range(0, 30):
    x.append(sides)
    x2.append(sides)
    y.append(prev_side_length * sides)
    y2.append(current * sides)
    prev_side_length = side_calculator(prev_side_length)
    current = new_side_calculator(current)
    sides *= 2
plt.plot(x, y, marker='o', label="f(x)")
plt.plot(x2, y2, marker='o', label="f(y)")
plt.axhline(2 * math.pi, linestyle="--", label="2π")
plt.xscale('log', base=2)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()

# a) Was passiert für grosse n und weshalb?
# The perimeter starts getting bigger 2π, reaching around 12, before dropping all the way to 0.

# b) Was beobachten Sie?
# The perimeter reaches 2π, but stays at that value, unlike the first method.