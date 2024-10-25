import cProfile
import numpy as np
import time
cProfile.run ('''

# Organizing Inputs
n = 1000
A = np.random.rand(n, n)  # Random n x n matrix
B = np.random.rand(n, n)  # Random n x n matrix
C = np.zeros((n, n))      # Preallocate a matrix for results
CC = np.zeros((n, n))     # Preallocate another matrix for results

# Triple nested loop for matrix multiplication (element-wise computation)
start_time_loop = time.perf_counter()
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
end_time_loop = time.perf_counter()
timeloop = end_time_loop - start_time_loop
print(f"Time for looped multiplication: {timeloop:.6f} seconds")

# Matrix multiplication using vectorized columns
start_time_loopvec = time.perf_counter()
for j in range(n):
    CC[:, j] = np.dot(A, B[:, j])
end_time_loopvec = time.perf_counter()
timeloopvec = end_time_loopvec - start_time_loopvec
print(f"Time for column-wise vectorized multiplication: {timeloopvec:.6f} seconds")

# Fully vectorized matrix multiplication
start_time_vec = time.perf_counter()
CCC = np.dot(A, B)
end_time_vec = time.perf_counter()
timevec = end_time_vec - start_time_vec
print(f"Time for fully vectorized multiplication: {timevec:.6f} seconds")

# Norm calculations
norm1 = np.linalg.norm(C - CC)
norm2 = np.linalg.norm(C - CCC)
print(f"Norm(C - CC): {norm1:.6f}")
print(f"Norm(C - CCC): {norm2:.6f}")

# Speedup calculations
Speedup = timeloop / timeloopvec
Speedup2 = timeloop / timevec
Speedup3 = timeloopvec / timevec
print(f"Speedup (loop vs vectorized columns): {Speedup:.6f}")
print(f"Speedup2 (loop vs fully vectorized): {Speedup2:.6f}")
print(f"Speedup3 (vectorized columns vs fully vectorized): {Speedup3:.6f}")
''')