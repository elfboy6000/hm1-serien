import numpy as np
import matplotlib.pyplot as plt

# 1) Definitionsbereich
x = np.linspace(a, b, 1000)  # z.B. a=-5, b=5

# 2) Funktionswerte
y = f(x)  # z.B. y = x**2 oder np.sin(x)

# 3) Plot erstellen
plt.figure()
# 3.5) label setzen
plt.plot(x, y, label="f(x) = ...")

# 4) Deko
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Graph von f")

# 5) Anzeigen / Speichern
plt.show()
