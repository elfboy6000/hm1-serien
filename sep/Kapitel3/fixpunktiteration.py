import sympy as sp


def fixpunktiteration(func_str: str, x0: float, tol: float = 1e-7, max_iter: int = 100) -> (str, str, str):
    x = sp.symbols('x')

    g = sp.sympify(func_str)

    # Calculate derivative and evaluate at starting point
    g_prime = sp.diff(g, x)
    g_prime_at_x0 = float(g_prime.evalf(subs={x: x0}))
    # Determine convergence based on |g'(x0)| < 1
    convergence_status = "Converges" if abs(g_prime_at_x0) < 1 else "Diverges"

    if convergence_status == "Converges":
        # Perform exactly max_iter iterations
        x_n = float(x0)
        for _ in range(max_iter):
            x_n = float(g.evalf(subs={x: x_n}))
            if abs(x_n) < tol:
                break

        ndigits = max(0, int(sp.ceiling(-sp.log(tol, 10))))
        x_n_rounded = round(x_n, ndigits)
        x_n_str = f"{g_prime_at_x0} < 1"
    else:
        x_n_str = f"{g_prime_at_x0} > 1"
        x_n_rounded = "N/A"

    return x_n_str, convergence_status, x_n_rounded


# Example usage:
print(fixpunktiteration('x**3+0.3', -1.125))  # Diverges
print(fixpunktiteration('x**3+0.3', 0.3389, 1e-5))  # Converges
print(fixpunktiteration('x**3+0.3', 0.7864))  # Diverges

print(fixpunktiteration('(x-0.3)**(1/3)', 0.7864))  # Converges

# Fixed: use E (Euler's number) or exp() function in sympy
print(fixpunktiteration('(((E**x)+(E**-x))/2)-(3/2)',  1.5, 1e-6))
