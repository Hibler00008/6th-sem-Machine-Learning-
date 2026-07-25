# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:25:34 2025

@author: shail
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_dims = 1000
probability_per_dim = 0.9
distance_per_dim = 3


dims =[]
joint_prob = []
euclidean_dist=[]
jp=1
my_dist = 0
for i in range(num_dims):
    jp = jp*probability_per_dim
    my_dist = np.sqrt(my_dist**2 + distance_per_dim**2)
    joint_prob.append(jp)
    euclidean_dist.append(my_dist)
    
    
dims = range(num_dims)

plt.scatter(dims, joint_prob)
plt.xlabel("#dimensions")
plt.ylabel("joint_prob")
plt.show()
 

plt.scatter(dims, np.log10(joint_prob))
plt.xlabel("#dimensions")
plt.ylabel("log scale joint_prob")
plt.show()
     
 

plt.scatter(dims, euclidean_dist)
plt.xlabel("#dimensions")
plt.ylabel("euclidean_dist")
plt.show()
    
