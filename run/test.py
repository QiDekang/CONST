import numpy as np
import pandas as pd
import random as rn
import os
import sys
'''
df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['A', 'B', 'C', 'D'])

print(df)
df = df.drop([0, len(df)-1])
print(df)
'''
'''
print(range(1, 1))

a = np.floor(10*np.random.random((6, 3, 5)))
print(a)
print(a[0, 0, 0])
print(a[:, 0, :])

print(a[:][0][:])

root_save_floder = 'D:/HeatingOptimization/STCDRank/result/model_result_0830/result_0902_window_size/'
windows_len = 2
folder_path = root_save_floder + str(windows_len) + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


import math
print(math.pow(2, 3))
print(math.pow(2, 1.16))
#print(math.pow(np.array([1,2,3,4]), 1.16))
print(np.array([1,2,3,4]) ** 2)
'''
file_name = 'dingfu_low'
file_list_one = [file_name]
print('file_list_one', file_list_one)
print('file_name', file_list_one[0])


for i in np.arange(0.90, 1.01, 0.01):
    #i = round(i, 2)
    print(i)


close_effect_rate_range = range(90, 101, 1)
for close_effect_id in close_effect_rate_range:
    close_effect_rate = round(0.01 * close_effect_id, 2)
    print('close_effect_rate',close_effect_rate)


for close_effect_id in np.arange(1, 5.5, 0.5):
    print(close_effect_id)