import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

def calculateSwaps(arr):
    temp_min = float('inf')
    var_change_count = 0
    for elt in arr:
        if elt<temp_min:
            temp_min = elt
            var_change_count+=1
    return var_change_count
        

def min_var_change_counts(K,N):
    #data = np.random.random(K*N).reshape(K,N)
    var_change_counts = []
    for it in range(K):
        arr1 = np.random.random(N)
        var_change_count = calculateSwaps(arr1)
        
        var_change_counts.append(var_change_count)
    return var_change_counts


K = 100
Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
var_change_counts_mean = []

for N in Ns:
    var_change_counts = min_var_change_counts(K, N)

    Mean     = np.mean(var_change_counts)
    Median   = np.median(var_change_counts)
    geoMean  = gmean(var_change_counts)

    
    print(f"N = {N} -> mean={Mean:.3f}, median={Median:.3f}, geo-mean={geoMean:.3f}")

    
    plt.figure()
    plt.hist(var_change_counts, bins=10)
    plt.title(f'K={K}, N={N}')
    plt.xlabel('var_change_count')
    plt.ylabel('Frequency')
    plt.show()
    var_change_counts_mean.append(np.mean(var_change_counts))


var_change_counts_mean = [float(val) for val in var_change_counts_mean]
plt.plot(Ns, var_change_counts_mean)
plt.xlabel('N')
plt.ylabel('mean_var_changed_count')
plt.show()
    
        