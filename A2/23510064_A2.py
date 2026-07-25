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


B = np.random.normal(5,2,10000)
I = plw.rvs(a=0.3,size=10000)
H = geom.rvs(p=0.005, size=10000)

plt.boxplot([B,I,H])
plt.title('Box Plot Comparison of B, I, and H')
plt.show()

def max_se_divide(B,I,H):
    B_naya = B/max(B)
    I_naya = I/max(I)
    H_naya = H/max(H)
    plt.hist(B,100,alpha=0.6,color='blue')
    plt.hist(B_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after Normalization by Max for B")
    plt.show()
    plt.hist(I,100,alpha=0.6,color='blue')
    plt.hist(I_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after Normalization by Max for I")
    plt.show()
    plt.hist(H,100,alpha=0.6,color='blue')
    plt.hist(H_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after Normalization by Max for H")
    plt.show()
    plt.boxplot([B_naya, I_naya, H_naya])
    plt.title('Comparison of Normalized Variables')
    plt.show()
    
def sum_se_divide(B,I,H):
    B_naya = B/sum(B)
    I_naya = I/sum(I)
    H_naya = H/sum(H)
    plt.hist(B,100,alpha=0.6,color='blue')
    plt.hist(B_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after Normalization by sum for B")
    plt.show()
    plt.hist(I,100,alpha=0.6,color='blue')
    plt.hist(I_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after Normalization by sum for I")
    plt.show()
    plt.hist(H,100,alpha=0.6,color='blue')
    plt.hist(H_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after Normalization by sum for H")
    plt.show()
    plt.boxplot([B_naya, I_naya, H_naya])
    plt.title('Comparison of Normalized Variables')
    plt.show()
   
def z_score(B,I,H):
    B_mean = np.mean(B)
    B_std = np.std(B)
    I_mean = np.mean(I)
    I_std = np.std(I)
    H_mean = np.mean(H)
    H_std = np.std(H)
    B_naya = (B-B_mean) / B_std
    I_naya = (I-I_mean) / I_std
    H_naya = (H-H_mean) / H_std
    plt.hist(B,100,alpha=0.6,color='blue')
    plt.hist(B_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after z-score Normalization for B")
    plt.show()
    plt.hist(I,100,alpha=0.6,color='blue')
    plt.hist(I_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after z-score Normalization for I")
    plt.show()
    plt.hist(H,100,alpha=0.6,color='blue')
    plt.hist(H_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after z-score Normalization for H")
    plt.show()
    plt.boxplot([B_naya, I_naya, H_naya])
    plt.title('Comparison of Normalized Variables')
    plt.show()
   
    
def percentile_n(B,I,H):
    B_ranks = rankdata(B, method='average') 
    B_naya = B_ranks/len(B)
    I_ranks = rankdata(I, method='average') 
    I_naya = I_ranks/len(I)
    H_ranks = rankdata(H, method='average') 
    H_naya = H_ranks/len(H)
    plt.hist(B,100,alpha=0.6,color='blue')
    plt.hist(B_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after percentile Normalization for B")
    plt.show()
    plt.hist(I,100,alpha=0.6,color='blue')
    plt.hist(I_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after percentile Normalization for I")
    plt.show()
    plt.hist(H,100,alpha=0.6,color='blue')
    plt.hist(H_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after percentile Normalization for H")
    plt.show()
    plt.boxplot([B_naya, I_naya, H_naya])
    plt.title('Comparison of Normalized Variables')
    plt.show()
 
    
def make_median_equal(B,I,H):
    B_med = np.median(B)
    I_med = np.median(I)
    H_med = np.median(H)
    m1 = (B_med+I_med+H_med)/3
    B_naya = B * (m1/B_med)
    I_naya = I * (m1/I_med)
    H_naya = H * (m1/H_med)
    #print(st.median(B_naya), st.median(I_naya), st.median(H_naya))
    plt.hist(B,100,alpha=0.6,color='blue')
    plt.hist(B_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after median alignment for B")
    plt.show()
    plt.hist(I,100,alpha=0.6,color='blue')
    plt.hist(I_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after median alignment for I")
    plt.show()
    plt.hist(H,100,alpha=0.6,color='blue')
    plt.hist(H_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after median alignment for H")
    plt.show()
    plt.boxplot([B_naya, I_naya, H_naya])
    plt.title('Comparison of Normalized Variables')
    plt.show()
    
    
def quantile_n(B,I,H):
    data = np.vstack([B, I, H]).T
    normalized_data = qt(data, output_distribution='normal', random_state=0, copy=True)
    B_naya = normalized_data[:, 0]
    I_naya = normalized_data[:, 1]
    H_naya = normalized_data[:, 2]
    plt.hist(B,100,alpha=0.6,color='blue')
    plt.hist(B_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after quantile Normalization for B")
    plt.show()
    plt.hist(I,100,alpha=0.6,color='blue')
    plt.hist(I_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after quantile Normalization for I")
    plt.show()
    plt.hist(H,100,alpha=0.6,color='blue')
    plt.hist(H_naya,100,alpha=0.6,color='orange')
    plt.title("Comparing after quantile Normalization for H")
    plt.show()
    plt.boxplot([B_naya, I_naya, H_naya])
    plt.title('Comparison of Normalized Variables')
    plt.show()
  
   
# max_se_divide(B,I,H)
# sum_se_divide(B,I,H)
# z_score(B,I,H  )
# make_median_equal(B,I,H)
# percentile_n(B,I,H)
# quantile_n(B,I,H)

