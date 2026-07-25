import numpy as np
import sys
import matplotlib.pyplot as plt

numbers = np.random.random(1000)
rnum = numbers.reshape(100, 10)

var_change_count_list = []
min_list = []

for arr1D in rnum:
    temp_min = sys.maxsize  
    var_change_count = 0
    for elt in arr1D:
        if elt<temp_min:  
            temp_min = elt
            var_change_count += 1
    var_change_count_list.append(var_change_count)
    min_list.append(temp_min)

plt.hist(var_change_count_list,10)
plt.show()