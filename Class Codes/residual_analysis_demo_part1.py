# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 15:48:48 2025

@author: shail
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# please note that a train test split is not 
# included as this  code aims to demonstrate
# residual analysis concepts  important for linea regression 

def  fit_and_perdict_linear_model(data, x_colname, y_colname):
    x=data[x_colname]
    y =data[y_colname]
    x = sm.add_constant(x)
    lr = sm.OLS(y, x) 
    result = lr.fit()
    print(result.summary())
    y_predict = result.predict(x)
    residual = y - y_predict
    data['fitted_y'] = y_predict
    data['Residual'] = residual
    plt.scatter(x.iloc[:,1].values,y )
    plt.plot(x.iloc[:,1],y_predict , color = 'red')
    plt.xlabel(x_colname)
    plt.ylabel(y_colname)
    plt.title("train fit")
    plt.show()   
    
    mean_residual = data.groupby(by='fitted_y')['Residual'].mean()
    plt.scatter(y_predict,residual )
    plt.plot(mean_residual, color= 'red')
    plt.xlabel('fitted')
    plt.ylabel('residual')
    plt.title("fitted_vs_residual")
    plt.show()  
    data['fitted_y'] = y_predict
    data['Residual'] = residual
    return data

######## part 1 ###########


# create a simple linear relationship

sample_size = 200
x = np.random.randint(1,10,sample_size)
noise = np.random.normal(10,2,sample_size)
y = 3.2*x + noise 
my_df = pd.DataFrame({"GPA":x, "income_lpa":y})
out_df = fit_and_perdict_linear_model(my_df, "GPA", 'income_lpa')


# introduce somewhat non_linearity 

y_non_linear = 0.5*x**2 + 3.2*x + noise
my_df = pd.DataFrame({"GPA":x, "income_lpa":y_non_linear})
out_df = fit_and_perdict_linear_model(my_df, "GPA", 'income_lpa')


#create y as funtion of two varaibles but use only one to  predict y 
family_wealth = np.random.normal(15,3,sample_size)
y_2_var_function = 3.2*x + 2.1*family_wealth + noise
my_df = pd.DataFrame({"GPA":x, "family_wealth":family_wealth,"income_lpa":y_2_var_function})

out_df = fit_and_perdict_linear_model(my_df, "GPA", 'income_lpa')

#let's check if family wealth exhibit any influence on residuals
plt.scatter(out_df['family_wealth'], out_df['Residual'])
plt.xlabel('family_wealth')
plt.ylabel('residual')
plt.title("family_wealth_vs_residual")
plt.show()





 

   

