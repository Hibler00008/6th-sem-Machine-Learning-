
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:22:30 2025

@author: shail
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split

f_heights = pd.Series( np.random.normal(152, 5, 1000))
m_heights =pd.Series (np.random.normal(166, 5, 1000))
num_females = f_heights.size
num_males = m_heights.size
total = num_females+ num_males

plt.hist([f_heights,m_heights], bins =100, label=['female','male'])
plt.legend(loc='upper right')
plt.show()





#@threshold_increment = 1
def threshold_classifier(threshold_increment ,f_heights, m_heights):
    lower_bound_of_overlap = m_heights.min()
    upper_bound_of_overlap = f_heights.max()
    total = f_heights.size + m_heights.size

    print(lower_bound_of_overlap,upper_bound_of_overlap)
    new_lower_bound = np.floor(lower_bound_of_overlap)
    new_upper_bound = np.ceil(upper_bound_of_overlap)
    
    current_min_miss_classification_rate = 100.0
    current_optimal_thershold = new_lower_bound
    
    
    for threshold in np.arange(new_lower_bound,new_upper_bound+1,threshold_increment ):
    
        misclassified_females = sum(f_heights>threshold)
        misclassified_males = sum(m_heights<threshold)
        misclassification_rate = 100.0*(misclassified_females+misclassified_males )/total
        print( threshold,misclassified_females, misclassified_males,misclassification_rate )
        print("currentmin",current_optimal_thershold )
        if misclassification_rate<current_min_miss_classification_rate:
           current_optimal_thershold = threshold
           current_min_miss_classification_rate = misclassification_rate
           print(" min changed", current_optimal_thershold) 
    return[current_optimal_thershold,current_min_miss_classification_rate ]
    


threshold_results = threshold_classifier(0.5,f_heights,m_heights)
print(threshold_results)
           



def probability_classifier(f_heights, m_heights):
    total = f_heights.size + m_heights.size

    male_mean = m_heights.mean()
    male_sd= m_heights.std()    
        
    female_mean = f_heights.mean()
    female_sd= f_heights.std() 
    
    print(male_mean, male_sd, female_mean, female_sd) 
    
    num_misclassified_females = 0
    for current_height in f_heights:
        female_probability = norm.pdf(current_height,female_mean,female_sd )
        male_probability = norm.pdf(current_height,male_mean,male_sd )
        
        if male_probability> female_probability:
            #print( "misclassified female",current_height,female_probability, male_probability)
            num_misclassified_females +=1
    
    print("female_misclassification",num_misclassified_females )
    num_misclassified_males = 0
    for current_height in m_heights:
        female_probability = norm.pdf(current_height,female_mean,female_sd )
        male_probability = norm.pdf(current_height,male_mean,male_sd )
        
        if male_probability < female_probability:
            #print("misclassified male",current_height,female_probability, male_probability)
            num_misclassified_males +=1
    print("male_misclassification",num_misclassified_males )
    
    misclassfication_rate = 100.0*(num_misclassified_females+num_misclassified_males )/total
    return misclassfication_rate


probability_misclassifiaction = probability_classifier(f_heights, m_heights)



heights = f_heights


def quantize(heights,interval_len):
    interval_label = np.floor( heights /interval_len)
    interval_counts = interval_label.value_counts()
    return interval_counts


#interval_len =0.5

def local_classifier(interval_len, f_heights, m_heights):
    total = f_heights.size + m_heights.size
    
    male_quantized = quantize(m_heights, interval_len)
    female_quantized = quantize(f_heights, interval_len)
    
    quantized_overlap_lower_bound = int(male_quantized.index.min())
    quantized_overlap_upper_bound = int(female_quantized.index.max())
    total_misclassification = 0
    map_list=[]
    for common_interval in range(quantized_overlap_lower_bound,quantized_overlap_upper_bound,1 ):
        
        female_count =  0 if common_interval not in female_quantized.index else female_quantized[common_interval]
        male_count =  0 if common_interval not in male_quantized.index else male_quantized[common_interval]
        error = min(female_count,male_count)
        total_misclassification+= error
        map_list.append({"interval":common_interval,'female_count':female_count,'male_count':male_count})
        #print( common_interval, female_count, male_count, error, total_misclassification)
    print("misclassification=" ,100.0*total_misclassification/total)
    interval_df=pd.DataFrame(map_list)
    return  interval_df



local_error = local_classifier(0.005,f_heights, m_heights)
#print(local_error)


