# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 23:43:45 2025

@author: ASUS
"""
import numpy as np 
import matplotlib.pyplot as plt

def sampling(arr):
    return np.random.choice(arr, 100)

B = np.random.randint(1,10000,1000)


var_store = []
s_var_store = []

for i in range (0,100):
    sample = sampling(B)
    mean = np.mean(sample)
    addition = 0

    for j in range (0,100):
        temp = pow((sample[j] - mean),2) 
        addition += temp

    s_variance = addition /( len(sample) - 5)
    variance = addition / len(sample)
    var_store.append(variance)
    s_var_store.append(s_variance)
    
p_var = []
main_var = np.var(B)
for i in range(0,100):
    p_var.append(main_var)


plt.plot(p_var, color='black')
plt.legend("pop var")
plt.plot(var_store, color = 'red')
plt.legend("sample var")
plt.plot(s_var_store, color = 'g')
plt.legend("sample var with c")
plt.show()


    


