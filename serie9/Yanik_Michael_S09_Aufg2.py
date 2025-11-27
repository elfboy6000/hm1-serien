import numpy as np

def Yanik_Michael_S09_Aufg2(A, Ae, b, eb):
    """
    Parameters
    ----------
    A : array_like
        Matrix des ursprünglichen LGS Ax = b.
    Ae : array_like
        Gestörte Matrix A~ des LGS A~ xe = eb.
    b : array_like
        Rechte Seite des ursprünglichen LGS.
    eb : array_like
        Gestörte rechte Seite des LGS.

    Returns
    -------
    x : ndarray
        Lösung des LGS Ax = b.
    xe : ndarray
        Lösung des gestörten LGS A~ xe = eb.
    dxmax : float
        Obere Schranke des relativen Fehlers gemäß
        dxmax = cond(A) / (1 - cond(A) * ||A - Ae|| / ||A||)
                * ( ||A - Ae|| / ||A|| + ||b - eb|| / ||b|| )
        (∞-Norm). Falls cond(A) * ||A - Ae|| / ||A|| >= 1:
        np.nan.
    dxobs : float
        Tatsächlicher relativer Fehler:
        dxobs = ||x - xe||_∞ / ||x||_∞
    """

    # In Arrays umwandeln
    A = np.asarray(A, dtype=float)
    Ae = np.asarray(Ae, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    eb = np.asarray(eb, dtype=float).reshape(-1)

    # Lösungen der LGS
    x = np.linalg.solve(A, b)
    xe = np.linalg.solve(Ae, eb)

    # Konditionszahl in ∞-Norm
    condA = np.linalg.cond(A, np.inf)

    # ∞-Normen
    normA = np.linalg.norm(A, np.inf)
    normA_diff = np.linalg.norm(A - Ae, np.inf)
    normb = np.linalg.norm(b, np.inf)
    normb_diff = np.linalg.norm(b - eb, np.inf)

    # Verhältnisse (relative Störungen)
    # Vorsicht, falls normA oder normb = 0 (dann dxmax bzw. dxobs nicht sinnvoll)
    if normA == 0 or normb == 0:
        dxmax = np.nan
    else:
        alpha = normA_diff / normA          # ||A - Ae|| / ||A||
        beta = normb_diff / normb          # ||b - eb|| / ||b||
        denom = 1.0 - condA * alpha        # 1 - cond(A) * alpha

        if denom <= 0:
            # Bedingung cond(A) * ||A-Ae|| / ||A|| < 1 nicht erfüllt
            dxmax = np.nan
        else:
            dxmax = condA / denom * (alpha + beta)

    # Beobachteter relativer Fehler
    normx = np.linalg.norm(x, np.inf)
    if normx == 0:
        dxobs = np.nan
    else:
        dxobs = np.linalg.norm(x - xe, np.inf) / normx

    return x, xe, dxmax, dxobs
