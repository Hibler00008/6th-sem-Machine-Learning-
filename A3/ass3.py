import numpy as np
import pandas as pd
from scipy.stats import powerlaw as plw
from scipy.stats import geom as geom
import matplotlib.pyplot as plt
import math
import scipy.stats
import statistics as st
from scipy.stats import rankdata
from sklearn.preprocessing import quantile_transform as qt


def z_score(B):
    B_mean = np.mean(B)
    B_std = np.std(B)
    B_naya = (B-B_mean) / B_std
    return B_naya

def robust(B):
    medB = np.median(B)
    q1B = np.percentile(B, 25)
    q3B = np.percentile(B, 75)
    iqrB = q3B - q1B
    if iqrB == 0:
        iqrB = 1.0
    return (B - medB) / iqrB

def add_outliers(arr):
    outliers = np.random.normal(1500, 500, 500) #5% outliers
    return np.concatenate([arr, outliers])

def plot_hist(clean, with_outliers, title):
    combined = np.concatenate([clean, with_outliers])
    low, high = np.percentile(combined, [1, 99])
    plt.hist(clean, bins=100, alpha=0.6, color='blue', label='Clean', range=(low, high))
    plt.hist(with_outliers, bins=100, alpha=0.6, color='orange', label='Outliers', range=(low, high))
    plt.title(title)
    plt.legend()
    plt.show()

B = np.random.normal(5, 8, 10000)
B_O = add_outliers(B)

S = np.random.lognormal(mean=5, sigma=1, size=10000)
S_O = add_outliers(S)

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
