import numpy as np
import matplotlib.pyplot as plt

from serie6.Yanik_Michael_S06_Aufg2 import Yanik_Michael_S06_Aufg2


def main():
    # ---------------------------------------------------------
    # Gegebene Daten
    # ---------------------------------------------------------
    years = np.array([1997, 1999, 2006, 2010], dtype=float)
    days = np.array([150, 104, 172, 152], dtype=float)

    # Zeitachse verschieben: 1997 -> 0, 1999 -> 2, 2006 -> 9, 2010 -> 13
    t = years - 1997.0

    # ---------------------------------------------------------
    # a) Polynom 3. Grades via LGS und Yanik_Michael_S06_Aufg2
    # p(t) = a * t^3 + b * t^2 + c * t + d
    # Wir stellen das 4x4-Vandermonde-System A * x = days auf.
    # ---------------------------------------------------------
    A = np.column_stack([t**3, t**2, t, np.ones_like(t)])  # Form (4,4)
    b_vec = days                                           # Form (4,)

    # Löse Ax = b mit deinem Gauss-Algorithmus
    U, detA, x = Yanik_Michael_S06_Aufg2(A, b_vec)

    # x enthält die Koeffizienten [a, b, c, d]
    a, b, c, d = x
    coeffs = x  # numpy-Array mit [a, b, c, d]

    print("=== Aufgabe 3a) Polynom 3. Grades (exakte Interpolation) ===")
    print(f"p(t) = {a:.5f} * t^3 + {b:.5f} * t^2 + {c:.5f} * t + {d:.5f}")

    # ---------------------------------------------------------
    # Plot: Datenpunkte + Polynomkurve (Original-Jahre auf x-Achse)
    # ---------------------------------------------------------
    years_fine = np.arange(1997, 2010.1, 0.1)
    t_fine = years_fine - 1997.0

    # numpy.polyval erwartet Koeffizienten [a, b, c, d] (höchster Grad zuerst)
    y_poly = np.polyval(coeffs, t_fine)

    plt.figure(figsize=(8, 5))
    # Datenpunkte
    plt.scatter(years, days, label="Messdaten", zorder=3)
    # Polynom
    plt.plot(years_fine, y_poly, label="Interpolationspolynom (3. Grad)", zorder=2)

    plt.xlabel("Jahr")
    plt.ylabel("Anzahl Tage mit extremer UV-Belastung")
    plt.title("UV-Tage in Hawaii – Interpolationspolynom 3. Grades")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ---------------------------------------------------------
    # b) Schätzwerte für 2003 und 2004
    # ---------------------------------------------------------
    year_2003 = 2003.0
    year_2004 = 2004.0
    t_2003 = year_2003 - 1997.0
    t_2004 = year_2004 - 1997.0

    estimate_2003 = np.polyval(coeffs, t_2003)
    estimate_2004 = np.polyval(coeffs, t_2004)

    print("\n=== Aufgabe 3b) Schätzwerte mit Interpolationspolynom ===")
    print(f"Schätzwert für 2003: {estimate_2003:.2f} Tage")
    print(f"Schätzwert für 2004: {estimate_2004:.2f} Tage")

    # ---------------------------------------------------------
    # c) Vergleich mit numpy.polyfit
    # ---------------------------------------------------------
    # polyfit berechnet die (least-squares-)Koeffizienten für ein Polynom 3. Grades
    coeffs_fit = np.polyfit(t, days, 3)  # [a_fit, b_fit, c_fit, d_fit]
    a_f, b_f, c_f, d_f = coeffs_fit

    print("\n=== Aufgabe 3c) Koeffizienten mit numpy.polyfit ===")
    print(f"p_fit(t) = {a_f:.5f} * t^3 + {b_f:.5f} * t^2 + {c_f:.5f} * t + {d_f:.5f}")

    estimate_2003_fit = np.polyval(coeffs_fit, t_2003)
    estimate_2004_fit = np.polyval(coeffs_fit, t_2004)

    print("\nSchätzwerte mit numpy.polyfit:")
    print(f"Schätzwert für 2003 (polyfit): {estimate_2003_fit:.2f} Tage")
    print(f"Schätzwert für 2004 (polyfit): {estimate_2004_fit:.2f} Tage")

    # Plot-Vergleich: exaktes Polynom vs. polyfit-Polynom
    y_poly_fit = np.polyval(coeffs_fit, t_fine)

    plt.plot(
        years_fine,
        y_poly_fit,
        "--",
        label="Polynom aus numpy.polyfit (3. Grad)",
        zorder=1,
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
