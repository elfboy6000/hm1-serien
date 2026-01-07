from itertools import chain
from math import sqrt

type Vector = list[float]
type Matrix = list[list[float]]


# --- Utils ---


def max_diff_mat(A: Matrix, B: Matrix) -> float:
  if dim(A) != dim(B):
    raise Exception("Both matrices must have same dimensions")
  largest_diff = 0
  for (a_row, b_row) in zip(A, B):
    diff = max_diff_vec(a_row, b_row)
    if diff > largest_diff:
      largest_diff = diff
  return largest_diff

def max_diff_vec(x1: Vector, x2: Vector) -> float:
  if len(x1) != len(x2):
    raise Exception("Both vectors must have same length")
  largest_diff = 0
  for (a, b) in zip(x1, x2):
    diff = abs(a - b)
    if diff > largest_diff:
      largest_diff = diff
  return largest_diff

def upper_triangle_matrix(A: Matrix) -> Matrix:
  n = len(A)
  for pivot_row in range(n):
    for row in range(pivot_row+1, n):
      factor = A[row][pivot_row] / A[pivot_row][pivot_row]
      for col in range(pivot_row, n):
        A[row][col] -= factor * A[pivot_row][col]
  return A

def gauss_algorithm(A: Matrix, b: Vector) -> (Matrix, Vector):
  n = len(A)
  for pivot_row in range(n):
    for row in range(pivot_row+1, n):
      factor = A[row][pivot_row] / A[pivot_row][pivot_row]
      for col in range(pivot_row, n):
        A[row][col] -= factor * A[pivot_row][col]
      b[row] -= factor * b[pivot_row]
  return (A, b)

def zero_mat(dimension: int) -> Matrix:
  return [[0] * dimension for _ in range(dimension)]

def id_mat(dimension: int) -> Matrix:
  I = zero_mat(dimension)
  # Converting diagonal elements to 1s
  for i in range(dimension):
    I[i][i] = 1
  return I

def trans(A: Matrix) -> Matrix:
  n = len(A)
  for i in range(n):
    for j in range(i+1):
      temp = A[i][j]
      A[i][j] = A[j][i]
      A[j][i] = temp
  return A

def dim(A: Matrix) -> (int, int):
  return (len(A), len(A[0]))

def trace(A: Matrix) -> float:
  Σ = 0
  n = len(A)
  for i in range(n):
    Σ += A[i][i]
  return Σ

def det(A: Matrix) -> float:
  A = upper_triangle_matrix(A)
  determinant = 1
  for i in range(0, len(A)):
      determinant *= A[i][i]
  return determinant

def norm(v: Vector) -> Vector:
  length = mag(v)
  # return list(map(lambda n: (n/length), v))
  return [(n/length) for n in v]

def mag(v: Vector) -> float:
  return sqrt(sum(map(lambda n: n**2, v)))

def mag2(v: Vector) -> float:
  return sum(map(lambda n: n**2, v))

def dot(A: Matrix, B: Matrix) -> Matrix:
  # Matrix dimensions have to be the same
  if len(A) != len(B) or len(A[0]) != len(B[0]):
    raise Exception("Incompatible dimensions for A({}) and B({})".format(len(A), len(B)))
  if len(A[0]) != len(B[0]):
    raise Exception("Incompatible dimensions for A[0]({}) and B[0]({})".format(len(A[0]), len(B[0])))
  for i in range(len(A)):
    for j in range(len(A[0])):
      A[i][j] *= B[i][j]
  return A

# TODO: use all variables
def matmul(A: Matrix, B: Matrix) -> Matrix:
  m, na = dim(A)
  nb, p = dim(B)
  # Matrix dimensions have to be compatible
  if len(A[0]) != len(B): return False
  # Matrices have dimensions A=m×n B=n×p
  n = len(A[0])
  m = len(A)
  p = len(B[0])
  C = [[0] * m for _ in range(p)]
  # Formula taken from [here](https://en.wikipedia.org/wiki/Matrix_multiplication#Definition)
  for i in range(m):
    for j in range(p):
      for k in range(n):
        C[i][j] += A[i][k]*B[k][j]
  return C

def mat_vec_to_vec(A: Matrix, v: Vector) -> Vector:
  x = []
  for row in A:
    new_x = 0
    for (v_val, a) in zip(v, row):
      new_x += v_val * a
    x.append(new_x)
  return x

def vec_vec_to_num(v: Vector) -> Vector:
  return sum([n**2 for n in v])

def vec_vec_to_num(v1: Vector, v2: Vector) -> Vector:
  return sum(map(lambda a,b: a*b, v1, v2))

def norm_1_vec(v: Vector) -> float:
  return abs(sum(v))

def norm_2_vec(v: Vector) -> float:
  return sqrt(sum(map(lambda n: n**2, v)))

def norm_inf_vec(v: Vector) -> float:
  return max(map(lambda n: abs(n), v))

def norm_inf_mat(A: Matrix) -> float:
  return max(map(lambda row: sum(map(lambda n: abs(n), row)), A))

def norm_1_mat(A: Matrix) -> float:
  A = trans(A)
  return norm_inf_mat(A)

def householder(v: Vector) -> Matrix:
  n = len(v)
  I = id_mat(n)
  v = norm(v)
  U = zero_mat(n)
  # U = [[0] * n for _ in range(n)]
  for i in range(n):
    for j in range(n):
      U[i][j] = 2*v[i]*v[j]
  for i in range(n):
    for j in range(n):
      I[i][j] -= U[i][j]
  return I

def expand_with_id(A: Matrix, size: int) -> Matrix:
  if len(A) != len(A[0]):
    raise Exception("Matrix must be square")
  if len(A) >= size:
    return A
  I = id_mat(size)
  offset = size-len(A)
  for i in range(len(A)):
    for j in range(len(A[0])):
      I[offset+i][offset+j] = A[i][j]
  return I

def print_matrix(name: str, A: Matrix, precision: int, final_new_line: bool):
  print('{}:'.format(name))
  for row in A:
    for value in row:
      print('{:> {width}.{precision}f} '.format(value, width=precision+1, precision=precision), end='')
    print()

  if final_new_line:
    print()

def __backwards_insertion(U: Matrix, y: Vector) -> Vector:
  """
  Solves the equation `Ux=y` after an LU decomposition.

  Returns x
  """
  n = len(y)
  x = [0] * n
  for i in range(n-1, -1, -1):
    Σx = 0
    for j in range(i,n):
      Σx += U[i][j]*x[j]
    x[i] = (1/U[i][i])*(y[i] - Σx)
  return x

def __forwards_insertion(L: Matrix, b: Vector) -> Vector:
  """
  Solves the equation `Ly=b` after an LU decomposition.

  Returns y
  """
  n = len(b)
  y = [0] * n
  for i in range(n):
    Σy = 0
    for j in range(i):
      Σy += L[i][j]*y[j]
    y[i] = b[i] - Σy
  return y


# --- End of utils ---


def newton(f, f1, x, tolerance):
  """`f1` has to be the first derivative of `f`, otherwise this won't work"""
  x_new = x+1
  while abs(x_new - x) >= tolerance:
    x = x_new
    x_new = x - (f(x)/f1(x))
  return x_new

def newton_iter(f, f1, x, iter_count):
  """`f1` has to be the first derivative of `f`, otherwise this won't work"""
  x_new = x+1
  for _ in range(iter_count):
    x = x_new
    x_new = x - (f(x)/f1(x))
  return x_new

def regula_falsi(f, x1, x2, tolerance):
  if f(x1)*f(x2) >= 0:
    return False
  while abs(x1 - x2) >= tolerance:
    xn = x2 - f(x2)*((x2-x1) / (f(x2)-f(x1)))
    x1 = x2
    x2 = xn
  return xn

def secant_method(f, x1, x2, tolerance):
  max_iter = 10000
  for _ in range(max_iter):
    f1 = f(x1)
    f2 = f(x2)
    denom = f2 - f1

    # es ist schwierig auf eine gute genauigkeit zu kommen, man muss das selbst definieren

    if abs(denom) < 1e-16:
      return None
    xn = x2 - f2 * (x2 - x1) / denom
    if abs(xn - x2) < tolerance:
      return xn
    x1, x2 = x2, xn
  return None

def regula_falsi_iter(f, x1, x2, iter_count):
  if f(x1)*f(x2) >= 0:
    return False
  for _ in range(iter_count):
    xn = x2 - f(x2)*((x2-x1) / (f(x2)-f(x1)))
    x1 = x2
    x2 = xn
  return xn

def binary_search_zero(f, x1, x2, tolerance):
  """
  Finds a zero of a function by performing binary search between the two given points (interval shrinking)
  """
  # Making sure that f(x1) < 0 and f(x2) > 0
  if f(x1) > 0 and f(x2) < 0:
    temp = x1
    x1 = x2
    x2 = temp
  # Each iteration halves the region of the solution until it's small enough
  while abs(f(x1) - f(x2)) >= tolerance:
    x_middle = (x1 + x2) / 2
    # Choosing the next point such that f(x1) < 0 < f(x2) is always true
    if f(x_middle) < 0:
      x1 = x_middle
    else:
      x2 = x_middle
  return x1

def binary_search_zero_iter(f, x1, x2, iter_count):
  """
  Finds a zero of a function by performing binary search between the two given points (interval shrinking)
  """
  # Making sure that f(x1) < 0 and f(x2) > 0
  if f(x1) > 0 and f(x2) < 0:
    temp = x1
    x1 = x2
    x2 = temp
  # Each iteration halves the region of the solution until it's small enough
  for _ in range(iter_count):
    x_middle = (x1 + x2) / 2
    # Choosing the next point such that f(x1) < 0 < f(x2) is always true
    if f(x_middle) < 0:
      x1 = x_middle
    else:
      x2 = x_middle
  return x1

def gauss(A: Matrix, b: Matrix) -> Vector:
  A, b = gauss_algorithm(A, b)
  x = [0] * len(b)
  n = len(A)
  # Backwards insertion (Ax=b)
  for i in range(n-1, -1, -1):
    sum = 0
    for j in range(i,n):
      sum += A[i][j]*x[j]
    x[i] = (1/A[i][i])*(b[i] - sum)
  return x

def gauss_jordan(A: Matrix, b: Vector) -> Vector:
  n = len(A)
  for pivot_row in range(n):
    # Normalise the entire row
    factor = A[pivot_row][pivot_row]
    for col in range(pivot_row, n):
      A[pivot_row][col] /= factor
    b[pivot_row] /= factor
    # Eliminate the column
    for row in range(n):
      if row == pivot_row:
        continue
      row_factor = A[row][pivot_row]
      for col in range(pivot_row, n):
        A[row][col] -= row_factor * A[pivot_row][col]
      b[row] -= row_factor * b[pivot_row]
  return b

def LR(A: Matrix) -> (Matrix, Matrix):
  return LU(A)

def LR_solve(A: Matrix, b: Vector) -> Vector:
  return LU_solve(A, b)

def LU(A: Matrix) -> (Matrix, Matrix):
  n = len(A)
  L = id_mat(n)
  U = A
  # LU decomposition
  for pivot_row in range(n):
    for row in range(pivot_row+1, n):
      factor = U[row][pivot_row] / U[pivot_row][pivot_row]
      L[row][pivot_row] = factor
      for col in range(pivot_row, n):
        U[row][col] -= factor * U[pivot_row][col]
  return (L, U)

def LU_solve(A: Matrix, b: Vector) -> Vector:
  L, U = LU(A)
  y = __forwards_insertion(L, b)
  return __backwards_insertion(U, y)

def QR(A: Matrix) -> (Matrix, Matrix):
  if len(A) != len(A[0]):
    raise Exception('Matrix is not square')
  n = len(A)
  Q = id_mat(n)
  R = A
  for col in range(n-1):
    a = list(chain(*map(lambda lst: lst[col:col+1], R)))[col:] # Copies the nth column below the diagonal from the matrix
    e = [0] * len(a) # Unit vector of same length
    e[0] = 1
    sig = 1 if a[0] >= 0 else -1
    mag_a = mag(a)
    v = list(map(lambda a, e: a + (sig*mag_a*e), a, e))
    H = householder(v)
    Qi = expand_with_id(H, n)
    R = matmul(Qi, R)
    Q = matmul(Q, trans(Qi))
  return (Q, R)

def QR_solve(A: Matrix,b: Vector) -> Vector:
  Q, R = QR(A)
  Q = trans(Q)
  # Matrix multiplying Q with b (Qb)
  b = mat_vec_to_vec(Q,b)
  return __backwards_insertion(R,b)

def jacobi(A: Matrix, b: Vector, x: Vector, tolerance: float) -> Vector:
  n = len(A)
  x_old = x.copy()
  x_new = [x+1 for x in x_old]
  while max_diff_vec(x_old, x_new) > tolerance:
    x_old = x_new.copy()
    # Jacobi
    for i in range(n):
      Σ = 0
      for j in range(n):
        if j == i:
          continue
        Σ += A[i][j]*x_old[j]
      x_new[i] = (1/A[i][i])*(b[i] - Σ)
    # print(x_new)
  return x_new

def jacobi_iter(A: Matrix, b: Vector, x: Vector, iteration_count: int) -> Vector:
  n = len(A)
  x_old = x.copy()
  x_new = [x+1 for x in x_old]
  for _ in range(iteration_count):
    x_old = x_new.copy()
    # Jacobi
    for i in range(n):
      Σ = 0
      for j in range(n):
        if j == i:
          continue
        Σ += A[i][j]*x_old[j]
      x_new[i] = (1/A[i][i])*(b[i] - Σ)
    # print(x_new)
  return x_new

def gauss_seidel(A: Matrix, b: Vector, x: Vector, tolerance: float) -> Vector:
  n = len(A)
  max_δ = 10*tolerance # In order to get the loop going
  while max_δ > tolerance:
    max_δ = 0 # The max diff is going to be new in each iteration
    for i in range(n): # Iterating over all elements of x
      Σ = 0
      for j in range(0, i): # Summing with the new xs
        Σ += A[i][j]*x[j]
      for j in range(i+1, n): # Summing with the old xs
        Σ += A[i][j]*x[j]
      new_x = (1/A[i][i])*(b[i] - Σ) # Calculating new x with Gauss-Seidel formula
      new_δ = abs(x[i] - new_x)
      if new_δ > max_δ: # Finding the largest diff
        max_δ = new_δ
      x[i] = new_x
  return x

def gauss_seidel_iter(A: Matrix, b: Vector, x: Vector, iteration_count: int) -> Vector:
  n = len(A)
  for _ in range(iteration_count):
    for i in range(n): # Iterating over all elements of x
      Σ = 0
      for j in range(0, i): # Summing with the new xs
        Σ += A[i][j]*x[j]
      for j in range(i+1, n): # Summing with the old xs
        Σ += A[i][j]*x[j]
      x[i] = (1/A[i][i])*(b[i] - Σ) # Calculating new x with Gauss-Seidel formula
  return x

def eigenvalues(A: Matrix, iter_count: int) -> Matrix:
  # P = id_mat(len(A))
  # while True:
  for _ in range(iter_count):
    Q, R = QR(A)
    A = matmul(R, Q)
    # P = matmul(P, Q)
  return A

def von_mises(A: Matrix, v: Vector, iter_count: int) -> (Vector, float):
  v = norm(v)
  for _ in range(iter_count):
    v = mat_vec_to_vec(A, v)
    norm_v = norm_2_vec(v)
    v = [n/norm_v for n in v]
  λ = (vec_vec_to_num(v, mat_vec_to_vec(A,v)))/(vec_vec_to_num(v,v))
  return (v, λ)

def vec_iter(A: Matrix, v: Vector, iter_count: int) -> (Vector, float):
  return von_mises(A, v, iter_count)

def power_method(A: Matrix, v: Vector, iter_count: int) -> (Vector, float):
  return von_mises(A, v, iter_count)
