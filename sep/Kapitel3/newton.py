# both newton and simplified newton method implementations
import sympy as sp
from typing import Callable, Union
from sekant import secant_method

X_SYMBOL = sp.symbols('x')
FunctionLike = Union[Callable[[float], float], sp.Expr]


def _as_callable(function: FunctionLike) -> Callable[[float], float]:
    if callable(function):
        return function
    return sp.lambdify(X_SYMBOL, function, 'math')


def newton_method(function: FunctionLike, x0: float, tol: float = 1e-7, max_iter: int = 100) -> float:
    func = _as_callable(function)
    derivative_expr = sp.diff(function, X_SYMBOL)
    derivative = sp.lambdify(X_SYMBOL, derivative_expr, 'math')

    x = float(x0)
    for iteration in range(1, max_iter + 1):
        f_x = func(x)
        f_prime_x = derivative(x)
        print(f"[Newton tol] Iter {iteration}: x = {x:.10f}, f(x) = {f_x:.3e}")
        if abs(f_x) < tol:
            return x
        if f_prime_x == 0:
            raise ValueError("Derivative is zero. No solution found.")
        x = x - f_x / f_prime_x

    raise ValueError("Maximum iterations reached. No solution found.")


def simplified_newton_method(function: FunctionLike, derivative: FunctionLike, x0: float, tol: float = 1e-7, max_iter: int = 100) -> float:
    func = _as_callable(function)
    derivative_callable = _as_callable(derivative)
    f_derv_x0 = derivative_callable(x0)

    if f_derv_x0 == 0:
        raise ValueError("Derivative is zero. No solution found.")

    x = float(x0)
    for iteration in range(1, max_iter + 1):
        f_x = func(x)
        print(f"[Simplified Newton] Iter {iteration}: x = {x:.10f}, f(x) = {f_x:.3e}")
        if abs(f_x) < tol:
            return x
        x = x - f_x / f_derv_x0

    raise ValueError("Maximum iterations reached. No solution found.")


def newton_method_fixed_iterations(function: FunctionLike, x0: float, iterations: int, true_root: float) -> tuple[float, float]:
    func = _as_callable(function)
    derivative_expr = sp.diff(function, X_SYMBOL)
    derivative = sp.lambdify(X_SYMBOL, derivative_expr, 'math')

    x = float(x0)
    for iteration in range(1, iterations + 1):
        f_x = func(x)
        f_prime_x = derivative(x)
        if f_prime_x == 0:
            raise ValueError("Derivative is zero. No solution found.")
        x_next = x - f_x / f_prime_x
        print(f"[Newton fixed] Iter {iteration}: x = {x_next:.10f}, f(x) = {func(x_next):.3e}")
        x = x_next

    true_root_val = float(true_root)
    if true_root_val == 0:
        rel_error = abs(x - true_root_val)
    else:
        rel_error = abs((x - true_root_val) / true_root_val)
    return x, rel_error

def run_demo() -> None:
    print("Example: Solve f(x) = x^2 - 2\n")
    function = X_SYMBOL**2 - 2
    derivative = sp.diff(function, X_SYMBOL)

    func = sp.lambdify(X_SYMBOL, function, 'math')
    derivative_callable = sp.lambdify(X_SYMBOL, derivative, 'math')
    true_root = float(sp.sqrt(2))

    print("1) Newton with tolerance-based stop:")
    tol_root = newton_method(function, x0=2.0, tol=1e-10, max_iter=10)
    print(f"Result: {tol_root:.10f}\n")

    print("2) Simplified Newton (derivative frozen at x0):")
    simplified_root = simplified_newton_method(func, derivative_callable, x0=1.0, tol=1e-6, max_iter=50)
    print(f"Result: {simplified_root:.10f}\n")

    print("3) Newton with fixed iteration count:")
    fixed_root, rel_error = newton_method_fixed_iterations(function, x0=1.0, iterations=4, true_root=true_root)
    print(f"Result after 4 iterations: {fixed_root:.10f}")
    print(f"Relative error vs sqrt(2): {rel_error:.3e}\n")

    print("4) Secant method:")
    secant_root = secant_method(function, x0=1.0, x1=2.0, tol=1e-10, max_iter=10)
    print(f"Result: {secant_root:.10f}\n")


run_demo()
