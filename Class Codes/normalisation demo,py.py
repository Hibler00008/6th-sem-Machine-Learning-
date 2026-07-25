# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:25:15 2025

@author: shail
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
num_samples = 100
#x = np.random.normal(2,4,num_samples)
#y = np.random.normal(100,20,num_samples)
#noise = np.random.normal(0,2,num_samples)
#y = 10*x + noise

#plt.figure(figsize=(10,10))
#plt.scatter(x,y)
#plt.show()



def generate_var_contribution_df(x,y, graph_title):
    dist_list =[]
    x_diff_list=[]
    y_diff_list=[]
    
    for i in range(num_samples):
        for j in range(num_samples):
            p1=[x[i],y[i]]
            p2=[x[j],y[j]]
            x_diff = np.abs((p1[0]-p2[0]))
            y_diff = np.abs((p1[1]-p2[1]))
            dist_sqr = x_diff**2 + y_diff**2
            dist_list.append(dist_sqr)
            x_diff_list.append(x_diff)
            y_diff_list.append(y_diff)
            
    df = pd.DataFrame({'x_diff':pd.Series(x_diff_list),
        'y_diff':pd.Series(y_diff_list),
         'dist_sqr':pd.Series(dist_list),
         }    )
        
    df['x_contribution'] = df.x_diff**2/ df.dist_sqr 
    df['y_contribution'] = df.y_diff**2/ df.dist_sqr 
    
    print (df.shape)
    df = df.dropna()
    print (df.shape)
    print(df.mean())
    plt.figure(figsize=(10,10))
    plt.hist([df['x_contribution'] , df['y_contribution']], label =['x_contri','yconti'])
    plt.legend(loc='upper right')
    plt.title(graph_title)
    plt.show()
    means = df.mean()
    print("x_contribution %:", 100.0*means['x_contribution'],
          "y_contribution:", 100.0*means['y_contribution'] )
    means['x_std']=np.std(x)
    means['y_std']= np.std(y)
    return  means


x = np.random.normal(0,1,num_samples)
out_map_list=[]
for sigma in [1,2,4,8,16,32,64]:
    y = np.random.normal(0,sigma,num_samples) 
    output=generate_var_contribution_df(x,y,"y_sigma:"+str(sigma)) 
    out_map_list.append(output)

result_df= pd.DataFrame(out_map_list)

result_df.plot(x='y_std',y=['x_contribution','y_contribution'])


