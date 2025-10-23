# Aufgabe S04_Aufg2 (Teil a)
# k_{i+1} = alpha * k_i * (1 - k_i), Startwert k0 = 0.1

def step(k, a):
    return a * k * (1.0 - k)

def analyze(alpha, k0=0.1, iters=50, tol=1e-4):
    k = k0
    seq = [k]
    for _ in range(iters):
        k = step(k, alpha)
        seq.append(k)

    # Zeige die letzten 10 Werte der Folge
    last = seq[-10:]
    print(f"\nalpha = {alpha}")
    print("letzte Werte:", ["{:.6f}".format(x) for x in last])

    # schauen, ob die letzten 5 Werte fast gleich sind
    last5 = seq[-5:]
    if max(last5) - min(last5) < tol:
        value = sum(last5) / len(last5)
        # Theoretische Fixpunkte
        fixpunktp0 = 0.0
        fixpunktp_nonzero = None
        if alpha != 0:
            fixpunktp_nonzero = 1.0 - 1.0 / alpha
        if abs(value - fixpunktp0) < 1e-3:
            print(f"-> Konvergiert gegen 0 (Infektion stirbt aus). (~{value:.6f})")
        elif fixpunktp_nonzero is not None and abs(value - fixpunktp_nonzero) < 1e-2:
            print(f"-> Konvergiert gegen nicht-null Fixpunkt ≈ {fixpunktp_nonzero:.6f} (gemessen ~{value:.6f})")
        else:
            print(f"-> Konvergiert gegen einen konstanten Wert ≈ {value:.6f}")
    else:
        print("-> Keine einfache Konvergenz (periodisch oder chaotisch)")

if __name__ == "__main__":
    alphas = [round(i * 0.1, 1) for i in range(0, 41)]
    for a in alphas:
        analyze(a)

# b)
# Das Modell beschriebt wie sich die Infektionsquote von Tag zu Tag entwickelt.
# Fixpunkte geben den Zustand, in dem sich wert k* einpendelt
# alpha-werte:
# - kleine alpha (bis 1) -> geht gegen 0, Krankheit verschwindet
# - mittlere alpha (von 1 bis 2) -> geht gegen einen konstanten nicht-null Fixpunkt, Krankheit bleibt, aber auf stabilem Niveau
# - größere alpha (von 2 bis 3) -> Periodizität oder Chaos, keine einfache Konvergenz, mal viele, mal wenige Infizierte
#
# realistische alpha-werte dürften im Bereich von 1 bis 2 liegen, da Krankheiten meist nicht sofort aussterben, aber auch nicht chaotisch verlaufen
# Aufgabe c) Siehe S04_Aufg1.pdf

