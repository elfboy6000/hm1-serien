import sympy as sp
from typing import Callable, Union

X_SYMBOL = sp.symbols('x')
FunctionLike = Union[Callable[[float], float], sp.Expr]

def _as_callable(function: FunctionLike) -> Callable[[float], float]:
    if callable(function):
        return function
    return sp.lambdify(X_SYMBOL, function, 'math')


def secant_method(function: FunctionLike, x0: float, x1: float, tol: float = 1e-7, max_iter: int = 100) -> float:
    func = _as_callable(function)

    x_prev = float(x0)
    x_curr = float(x1)

    for iteration in range(1, max_iter + 1):
        f_prev = func(x_prev)
        f_curr = func(x_curr)
        print(f"[Secant] Iter {iteration}: x = {x_curr:.10f}, f(x) = {f_curr:.3e}")
        if abs(f_curr) < tol:
            return x_curr
        if f_curr - f_prev == 0:
            raise ValueError("Division by zero in secant method. No solution found.")
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        x_prev, x_curr = x_curr, x_next

    raise ValueError("Maximum iterations reached. No solution found.")

def run_demo() -> None:
    print("Example: Solve f(x) = x^2 - 2\n")
    function = X_SYMBOL**2 - 2

    print("Secant method:")
    secant_root = secant_method(function, x0=1.0, x1=2.0, tol=1e-10, max_iter=10)
    print(f"Result: {secant_root:.10f}\n")


run_demo()