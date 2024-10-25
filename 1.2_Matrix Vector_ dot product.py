import numpy as np
import time

# Start fresh by generating random matrices and vectors
n = 100
A = np.random.rand(n, n)  # Random n x n matrix
x = np.random.rand(n, 1)  # Random n x 1 vector
b = np.zeros((n, 1))      # Preallocate n x 1 result vector
bb = np.zeros((n, 1))     # Preallocate n x 1 result vector

# Loop-based matrix-vector multiplication
start = time.perf_counter()
for i in range(n):
    for j in range(n):
        b[i] += A[i, j] * x[j]
timeloop = time.perf_counter() - start
print(f"Time for looped multiplication: {timeloop:.6f} seconds")

# Row-wise vectorized multiplication
start = time.perf_counter()
for i in range(n):
    bb[i] = np.dot(A[i, :], x)
timeloopvec = time.perf_counter() - start
print(f"Time for row-wise vectorized multiplication: {timeloopvec:.6f} seconds")

# Fully vectorized matrix-vector multiplication
start = time.perf_counter()
bbb = np.dot(A, x)
timevec = time.perf_counter() - start
print(f"Time for fully vectorized multiplication: {timevec:.6f} seconds")

# Norm calculations to check accuracy
norm_b_bb = np.linalg.norm(b - bb)
norm_b_bbb = np.linalg.norm(b - bbb)
print(f"Norm of (b - bb): {norm_b_bb}")
print(f"Norm of (b - bbb): {norm_b_bbb}")

# Speedup calculations
Speedup = timeloop / timeloopvec
Speedup2 = timeloop / timevec
Speedup3 = timeloopvec / timevec
print(f"Speedup (loop vs row-wise vectorized): {Speedup:.6f}")
print(f"Speedup (loop vs fully vectorized): {Speedup2:.6f}")
print(f"Speedup (row-wise vs fully vectorized): {Speedup3:.6f}")
