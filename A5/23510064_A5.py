
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(8)
dataset_size = 1000
x = np.random.normal(2, 5, dataset_size)

b0_true = 5.0
b1_true = 3.0
y_dash = x * b1_true + b0_true

noise_stds = [1,5,10,50,100,500,1000,5000,10000]
print(noise_stds)

b0_list = []
b1_list = []
R2_list = []
b0_p_value = []
b1_p_value = []
model_p_value = []

for std in noise_stds:
    noise = np.random.normal(0, std, dataset_size)
    y = y_dash + noise 
    
    X = sm.add_constant(x)         
    model = sm.OLS(y, X).fit()
    
    b0_list.append(model.params[0])       
    b1_list.append(model.params[1])      
    R2_list.append(model.rsquared)
    b0_p_value.append(model.pvalues[0])
    b1_p_value.append(model.pvalues[1])
    model_p_value.append(model.f_pvalue)

df = pd.DataFrame({
    "noise_std": noise_stds,
    "b0_estimated": b0_list,
    "b1_estimated": b1_list,
    "b0_pvalue": b0_p_value,
    "b1_pvalue": b1_p_value,
    "R2": R2_list,
    "model_pvalue": model_p_value
})
# print(df.round(6)

plt.plot(noise_stds, b0_list, marker='o')
plt.xlabel("Noise std")
plt.ylabel("Estimated b0 (intercept)")
plt.title("Effect of noise on estimated intercept (b0)")
plt.grid(True)
plt.show()

plt.plot(noise_stds, b1_list, marker='o')
plt.axhline(b1_true, color='red', linestyle='--', label='True b1')
plt.xlabel("Noise std")
plt.ylabel("Estimated b1 (slope)")
plt.title("Effect of noise on estimated slope (b1)")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(noise_stds, R2_list, marker='o')
plt.xlabel("Noise std")
plt.ylabel("R2")
plt.title("Effect of noise on R²")
plt.grid(True)
plt.show()

plt.plot(noise_stds, b1_p_value, marker='o')
plt.yscale('log')   
plt.xlabel("Noise std")
plt.ylabel("p-value of slope (b1)")
plt.title("Effect of noise on p-value of slope")
plt.grid(True)
plt.show()

plt.plot(noise_stds, model_p_value, marker='o')
plt.yscale('log')
plt.xlabel("Noise std")
plt.ylabel("model p-value")
plt.title("Effect of noise on model p-value")
plt.grid(True)
plt.show()


# =============================================================================
#  -sample size of 1000, the results suggest that larger sample sizes would be 
#   beneficial for noisy datasets
#  -if we lower the sample size, effect of noise is significant
#  -with increasing std, R2 decreases
#  -with increasing std, p value of the model increases  
#  -even with increasing p value the linear regression maintains
#   remarkable robustness to noise contamination with certain
#   performance degradation
#  -null hypotheses is that there is no relationship btw X & y(b1=0)
#  -p_value>0.05, we accept the null hypothesis
#  -f-statistics: model
#  -t-statistics: a particular variable
# =============================================================================
