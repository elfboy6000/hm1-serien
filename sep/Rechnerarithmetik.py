# Rechnerarithmetik

# --- IEEE-754 binary32 environment + decoder (base 2, 1 sign, 8 exp, 23 frac) ---

from dataclasses import dataclass


@dataclass(frozen=True)
class IEEEEnv:
    base: int = 2
    exp_bits: int = 8
    frac_bits: int = 23

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1


BINARY32 = IEEEEnv()


def _frac_field_to_fraction(frac_field: int, frac_bits: int, base: int = 2) -> float:
    """
    Interpret frac_field as a fixed-width fraction field (bits after the point).
    For binary32: value = sum(bit_i * 2^-(i+1)) for i=0..22.
    """
    if frac_field < 0 or frac_field >= (1 << frac_bits):
        raise ValueError("frac_field out of range for given frac_bits")

    frac = 0.0
    for i in range(frac_bits):
        bit = (frac_field >> (frac_bits - 1 - i)) & 1
        if bit:
            frac += base ** (-(i + 1))
    return frac


def ieee754_value(env: IEEEEnv, sign: int, exp_field: int, frac_field: int) -> float:
    """
    Decode IEEE-754 value from fields (not from Python's literal formatting).
    """
    if sign not in (0, 1):
        raise ValueError("sign must be 0 or 1")
    if exp_field < 0 or exp_field >= (1 << env.exp_bits):
        raise ValueError("exp_field out of range")
    if frac_field < 0 or frac_field >= (1 << env.frac_bits):
        raise ValueError("frac_field out of range")

    frac = _frac_field_to_fraction(frac_field, env.frac_bits, env.base)

    if exp_field == 0:
        # zero or subnormal
        if frac_field == 0:
            return -0.0 if sign else 0.0
        significand = 0.0 + frac
        exponent = 1 - env.bias
    elif exp_field == (1 << env.exp_bits) - 1:
        # inf / nan (optional handling)
        if frac_field == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")
    else:
        # normal
        significand = 1.0 + frac
        exponent = exp_field - env.bias

    return (-1.0 if sign else 1.0) * significand * (env.base ** exponent)


# --- Your inserts (binary32 fields) ---

print("2.1.3 (IEEE-754 binary32)")

# a) 2.0  => sign=0, exp=128, frac=0
print("a)", ieee754_value(BINARY32, sign=0, exp_field=128, frac_field=0))  # 2.0

# b) 6.5 => 1.101 * 2^2 => exp=127+2=129, frac bits "101000..0" => 0b101 << 20
print("b)", ieee754_value(BINARY32, sign=0, exp_field=129, frac_field=(0b101 << 20)))  # 6.5

# c) -6.5
print("c)", ieee754_value(BINARY32, sign=1, exp_field=129, frac_field=(0b101 << 20)))  # -6.5

# d) smallest normal: 2^-126 => exp=1, frac=0
print("d)", ieee754_value(BINARY32, sign=0, exp_field=1, frac_field=0))  # 2**-126

# e) subnormal with fraction field = 0.1 (i.e., top fraction bit set) => frac_field = 1<<22
# value = 2^-126 * 2^-1 = 2^-127
print("e)", ieee754_value(BINARY32, sign=0, exp_field=0, frac_field=(1 << 22)))  # 2**-127

# f) smallest subnormal: frac_field=1 => 2^-126 * 2^-23 = 2^-149
print("f)", ieee754_value(BINARY32, sign=0, exp_field=0, frac_field=1))  # 2**-149
