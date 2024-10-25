import cProfile
import numpy as np
import time
cProfile.run ('''

# initializing
n = 10000  # fix the length of the vectors
a = np.random.rand(n)  # create a random array a
b = np.random.rand(n)  # create a random array b

# dot product with for loop
c = 0  # defining a variable to store product result
start_time = time.time()  # storing the time in float value to a variable
for i in range(n):
    c += a[i] * b[i]  # adding up and storing the product value in c
timeloop = time.time() - start_time  # measure the time taken for the loop
print("Time taken with for loop:", timeloop)
print("Result with for loop:", c)

# dot product with vectorization
start_time = time.time()  # replacing the value

cc = np.dot(a, b)  # vectorized dot product

timevec = time.time() - start_time  # measuring the time taken for vectorization
print("Time taken with vectorization:", timevec)
print("Result with vectorization:", cc)

# Comparing both the results
print("Difference between results:", np.abs(c - cc))

# calculation speed comparison
speedup = timeloop / timevec
print("Speed-up:", speedup)
''')