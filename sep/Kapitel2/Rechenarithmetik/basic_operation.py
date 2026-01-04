# --- Wertebereich ---

def find_x_min(base, exponent_min):
    return base ** (exponent_min - 1)


def find_x_max(base, exponent_max, mantissa_length):
    return (1 - base ** (-mantissa_length)) * (base ** exponent_max)

def amount_of_representable_numbers_without_zero(base, mantissa_length, exponent_length):
    num_mantissas = base ** mantissa_length
    num_exponents = base ** exponent_length
    return num_mantissas * num_exponents

# --- Fehlerabschätzung ---

def absolute_error(x_approx, x_exact):
    return abs(x_approx - x_exact)

def relative_error(x_approx, x_exact):
    return abs(x_approx - x_exact) / abs(x_exact)

# don't forget hidden bit in the mantissa
def machine_accuracy(base, mantissa_length):
    return 0.5 * (base ** (1-mantissa_length))

def machine_accuracy_your_PC():
    eps = 1
    while 1. + eps > 1:
        eps = eps / 2
    return eps

