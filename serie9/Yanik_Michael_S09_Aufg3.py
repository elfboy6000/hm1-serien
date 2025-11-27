import numpy as np
import matplotlib.pyplot as plt

# Wir importieren die Funktion aus Aufgabe 2:
# Sie muss in der Datei Yanik_Michael_S09_Aufg2.py liegen.
from serie9.Yanik_Michael_S09_Aufg2 import Yanik_Michael_S09_Aufg2


def main():
    # Anzahl der Versuche
    n_iter = 1000

    # Vektoren für dxmax, dxobs und Verhältnis dxmax/dxobs
    dxmax_vals = np.zeros(n_iter)
    dxobs_vals = np.zeros(n_iter)
    ratio_vals = np.zeros(n_iter)

    # Für Reproduzierbarkeit (optional)
    np.random.seed(0)

    for k in range(n_iter):
        # Zufällige 100x100 Matrix A und 100x1 Vektor b
        A = np.random.rand(100, 100)
        b = np.random.rand(100, 1)

        # Gestörte Matrix A~ und gestörter Vektor b~
        Ae = A + np.random.rand(100, 100) / 1e5
        eb = b + np.random.rand(100, 1) / 1e5

        # Aufruf der Funktion aus Aufgabe 2
        # Name_S09_Aufg2(A, Ae, b, eb) gibt (x, xe, dxmax, dxobs)
        _, _, dxmax, dxobs = Yanik_Michael_S09_Aufg2(A, Ae, b, eb)

        dxmax_vals[k] = dxmax
        dxobs_vals[k] = dxobs

        # Verhältnis dxmax / dxobs
        if np.isnan(dxmax) or np.isnan(dxobs) or dxobs == 0:
            ratio_vals[k] = np.nan
        else:
            ratio_vals[k] = dxmax / dxobs

    # Grafische Darstellung mit semilogy
    it = np.arange(1, n_iter + 1)

    plt.figure()
    plt.semilogy(it, dxmax_vals, label=r"$\delta x_{\max}$")
    plt.semilogy(it, dxobs_vals, label=r"$\delta x_{\mathrm{obs}}$")
    plt.semilogy(it, ratio_vals, label=r"$\delta x_{\max} / \delta x_{\mathrm{obs}}$")
    plt.xlabel("Iteration")
    plt.ylabel("Wert (log-Skala)")
    plt.title(r"Vergleich von $\delta x_{\max}$ und $\delta x_{\mathrm{obs}}$")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.show()

    # Kurzer Kommentar im Skript (Aufgabe verlangt Kommentar im Code):
    """
    Kommentar:
    In dieser Versuchsreihe ist dx_max in der Regel eine gültige, aber oft
    sehr konservative (d.h. viel zu große) obere Schranke für dx_obs.
    Durch die relativ kleine Störung (Größenordnung 10^{-5}) und die zufälligen
    Matrizen ist der tatsächliche Fehler dx_obs oft deutlich kleiner als dx_max.
    Damit ist dx_max theoretisch korrekt, aber in der Praxis als Schätzung
    häufig recht pessimistisch.
    """


if __name__ == "__main__":
    main()
