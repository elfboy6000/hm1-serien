import numpy as np

# Gauss-Algorithmus mit partieller Pivotisierung
# Parameter:
#   A : quadratische Matrix (array-like, Form (n,n))
#   b : rechte Seite (array-like, Länge n)
#   eps : kleine Schwelle für Vergleich mit Null (numerische Toleranz)
def Yanik_Michael_S06_Aufg2(A, b, eps=1e-12):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Prüfungen auf Dimensionen
    # ndim == 1 → Vektor; ndim == 2 → Matrix; ndim > 2 → Tensor.
    # A.shape[0] → Anzahl Zeilen; A.shape[1] → Anzahl Spalten
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A muss eine quadratische Matrix sein.")
    n = A.shape[0]
    if b.shape != (n,) and b.shape != (n,1):
        raise ValueError("b muss die Länge n haben (Vektor).")
    b = b.reshape(n)  # 1-Dim Array für b
    # shape überprüft nur Form, nicht Werte

    # U = obere Dreiecksmatrix, 
    # # rhs = veränderte rechte Seite
    U = A.copy()
    rhs = b.copy()

    # Anzahl der Zeilenvertauschungen
    det_sign = 1

    # Gauss-Elimination mit partieller Pivotisierung
    # für i = 1...n-1 (Index k = 0..n-1)
    for k in range(n):

        # Finde in Spalte k ab Zeile k die Zeile mit grösstem Absolutwert als Pivot
        pivot_row = np.argmax(np.abs(U[k:n, k])) + k
        pivot_val = U[pivot_row, k]

        if abs(pivot_val) < eps:
            # Determinante 0
            raise ValueError("Matrix ist singulär oder nahezu singulär. Keine eindeutige Lösung.")

        # Vertausche Zeilen pivot_row <-> k wenn Pivot nicht in Zeile k drin
        if pivot_row != k:
            # Zeilen in U und rhs vertauschen
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]
            # Jede Zeilentausch ändert Vorzeichen der Determinante
            det_sign *= -1

        # Nun ist U[k,k] der Pivot. Eliminiere Einträge unterhalb der Pivot in Spalte k.
        pivot = U[k, k]
        # Für jede Zeile i unterhalb k: Faktor = U[i,k] / pivot
        # dann Zeile_i = Zeile_i - Faktor * Zeile_k
        for i in range(k + 1, n):
            factor = U[i, k] / pivot
            U[i, k:n] = U[i, k:n] - factor * U[k, k:n]
            rhs[i] = rhs[i] - factor * rhs[k]

    # Nach der Eliminationsschleife ist U obere Dreiecksmatrix
    A_triangle = U.copy()

    # Bestimme Determinante: Produkt der Diagonalelemente * Vorzeichen aus Zeilentauschen
    diag_prod = np.prod(np.diag(A_triangle))
    detA = det_sign * diag_prod

    # Wenn Determinante 0
    if abs(detA) < eps:
        raise ValueError("Determinante ist null, kein eindeutiges Lösungsergebnis.")

    # Rückwärtseinsetzen zur Bestimmung von x 
    x = np.zeros(n, dtype=float)
    # Beginne bei der letzten Zeile und arbeite nach oben
    for i in range(n - 1, -1, -1):
        # Summand der bereits bekannten x-Werte: sum_{j=i+1..n-1} U[i,j] * x[j]
        if i < n - 1:
            sum_known = np.dot(U[i, i+1:n], x[i+1:n])
        else:
            sum_known = 0.0
        # Stelle sicher, dass Diagonalelement nicht Null ist
        if abs(U[i, i]) < eps:
            raise ValueError("Null auf Diagonale während Rückwärtseinsetzen: Matrix singulär.")
        x[i] = (rhs[i] - sum_known) / U[i, i]

    # Rückgabe: obere Dreiecksmatrix, Determinante und Lösung
    return A_triangle, detA, x

# Test der Funktion mit Beispiel
A_ex = [[2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0]]
b_ex = [8.0, -11.0, -3.0]

U, detA, x = Yanik_Michael_S06_Aufg2(A_ex, b_ex)
print("Obere Dreiecksmatrix U:\n", U)
print("Determinante det(A) =", detA)
print("Lösung x =", x)
