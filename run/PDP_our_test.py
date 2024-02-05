#  跨文件调用
import os
import sys
current_path=os.getcwd().replace("\\","\\\\")
#current_path=os.path.dirname(os.getcwd())
sys.path.append(current_path)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from pandas import DataFrame

from common.config import PDP_grid_resolution

data = pd.read_csv('D:/HeatingOptimization/STCD_2023/data/data_preparation/all_sensors_average/qintaoyuan_low_data.csv')

max_value = data['second_heat_temp'].max()
min_value = data['second_heat_temp'].min()
print(max_value)
print(min_value)

grid_value = np.linspace(min_value, max_value, PDP_grid_resolution)
#print(grid_value)
grid_value = np.around(grid_value, 3)
print(grid_value)
print(grid_value[0])
print(len(grid_value))

