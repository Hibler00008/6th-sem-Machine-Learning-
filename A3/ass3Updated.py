import numpy as np
import pandas as pd
from scipy.stats import powerlaw as plw
from scipy.stats import geom as geom
import matplotlib.pyplot as plt
import math
import scipy.stats
import statistics as st
from scipy.stats import rankdata

def add_outliers_arr(arr):
    outliers = np.random.normal(1500, 500, 50) #5% outliers
    return np.concatenate([arr, outliers])

def cal_mean(arr):
    return np.mean(arr)

def cal_med(arr):
    return np.median(arr)

def cal_sd(arr):
    return np.std(arr)

def cal_iqr(arr):
    q1 = np.percentile(arr, 25, method='nearest')
    q3 = np.percentile(arr, 75, method='nearest')
    return q3-q1
    
def z_score(B):
    B_mean = np.mean(B)
    B_std = np.std(B)
    B_naya = (B-B_mean) / B_std
    return B_naya

def robust(B):
    medB = np.median(B)
    q1B = np.percentile(B, 25, method="nearest")
    q3B = np.percentile(B, 75, method="nearest")
    iqrB = q3B - q1B
    if iqrB == 0:
        iqrB = 1.0
    return (B - medB) / iqrB

def add_outliers(arr, outliers):
    return np.concatenate([arr, outliers])

def plot_hist(clean, with_outliers, title):
    combined = np.concatenate([clean, with_outliers])
    low, high = np.percentile(combined, [1, 99])
    plt.hist(clean, bins=100, alpha=0.6, color='blue', label='Clean', range=(low, high))
    plt.hist(with_outliers, bins=100, alpha=0.6, color='orange', label='Outliers', range=(low, high))
    plt.title(title)
    plt.legend()
    plt.show()
    
mean_store = []
med_store = []
sd_store = []
iqr_store = []

    
def one_by_one_outlier(arr, outliers, i):
    arr = np.append(arr, outliers[i])
    print("Mean after adding", i+1, "outliers", cal_mean(arr))
    print("Median after adding", i+1, "outliers", cal_med(arr))
    print("SD after adding", i+1, "outliers",cal_sd(arr))
    print("IQR after adding", i+1, "outliers",cal_iqr(arr))
    print()
    mean_store.append(cal_mean(arr))
    med_store.append(cal_med(arr))
    sd_store.append(cal_sd(arr))
    iqr_store.append(cal_iqr(arr))
    return arr
    
outliers = np.random.normal(150, 5, 50) #5% outliers

B = np.random.normal(5, 8, 1000)
B_O = B

S = np.random.lognormal(mean=5, sigma=1, size=1000)
S_O = add_outliers_arr(S)

print("Mean after adding", 0, "outliers", cal_mean(B))
print("Median after adding", 0, "outliers", cal_med(B))
print("SD after adding", 0, "outliers",cal_sd(B))
print("IQR after adding", 0, "outliers",cal_iqr(B))
print()
mean_store.append(cal_mean(B))
med_store.append(cal_med(B))
sd_store.append(cal_sd(B))
iqr_store.append(cal_iqr(B))
 
i = 0 
while i<50:
    B_O = one_by_one_outlier(B_O, outliers, i)
    i+=1
    
# plt.hist(mean_store, 100)
# plt.show()
# plt.hist(med_store, 100)
# plt.show()
# plt.hist(sd_store, 100)
# plt.show()
# plt.hist(iqr_store, 100)
# plt.show()

plt.plot(mean_store, color='r')
plt.plot(med_store, color='g')
plt.plot(sd_store, color='skyblue')
plt.plot(iqr_store, color = 'black')
plt.title("mean: red, med: green, sd: skyblue, iqr: black")
plt.show()

# normalizations:
# z-score
B_znorm = z_score(B)
BO_znorm = z_score(B_O)
S_znorm = z_score(S)
SO_znorm = z_score(S_O)

# robust scaling
B_rnorm = robust(B)
BO_rnorm = robust(B_O)
S_rnorm = robust(S)
SO_rnorm = robust(S_O)

plot_hist(B_znorm, BO_znorm, "z-score scaling (Normal data)")
plot_hist(B_rnorm, BO_rnorm, "robust scaling (Normal data)")
plot_hist(S_znorm, SO_znorm, "z-score scaling (Skewed data)")
plot_hist(S_rnorm, SO_rnorm, "robust scaling (Skewed data)")



