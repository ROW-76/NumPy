import numpy as np
import time
from joblib import Parallel, delayed
from scipy.optimize import fsolve

# Function to find the root using numpy for array handling
def findroot(x) :
    f =np.sin(3 * np.pi * np.cos(2 * np.pi * x) * np.sin(np.pi * x))
    return f

a = -3
b = 5
n = 4 ** 9
result=[]

x0 = np.linspace(a, b, n)

# Series for loop
series_time = time.perf_counter()
result_series = [fsolve(findroot, x) for x in x0]
total_s_time = time.perf_counter() - series_time
print("Series time:", total_s_time)

# Parallel Computation
num_workers=4
parallel_time = time.perf_counter()
result_parallel = Parallel(n_jobs=num_workers)(delayed(fsolve)(findroot, x) for x in x0)
total_p_time = time.perf_counter() - parallel_time
print("Parallel time:", total_p_time)

# Calculating Speedup and Efficiency
Speedup = total_s_time / total_p_time
Efficiency = Speedup / num_workers*100  # Adjusted to the number of parallel jobs
print("Speedup:", Speedup)
print("Efficiency(%):", Efficiency)