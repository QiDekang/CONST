import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from pandas import DataFrame

#data = pd.read_csv('D:/HeatingOptimization/STCD_2023/data/data_preparation/all_sensors_average/qintaoyuan_low_data.csv')
range_list = range(0, 100, 1)
print(range_list)
range_list = list(range_list)
print(range_list)
direction_data = pd.DataFrame(range_list, columns=["diff_heat"])
print(direction_data)

direction_data = pd.DataFrame([-2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], columns=["diff_heat"])
print(direction_data)