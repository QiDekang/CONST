import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from pandas import DataFrame
sns.set(style="ticks")
plt.figure(figsize=(8,6))
sns.set_context('notebook', font_scale=1.5)
x = range(0, 120, 1)
print(x)
xx = [0, 30, 60, 90]
print(xx)
dates = pd.date_range("11 15 2018", periods=120, freq="D")
print(dates[0])
print(dates[30])
print(dates[61])
print(dates[92])
print(dates[119])
#rootFolder = 'C:\\HeatingOptimization\\code\\plot\\'
#values = np.loadtxt(rootFolder + 'values.csv', delimiter=',', dtype='float')
#data = pd.DataFrame(values, x, columns=["HeatT", "IndoorT", "OutoorT"])
data = pd.read_csv('D:/HeatingOptimization/STCD_2023/data/data_preparation/all_sensors_average/qintaoyuan_low_data.csv')
data = data[['indoor_temp', 'second_heat_temp', 'outdoor_temp']]
plt.xticks( (0, 30, 61, 92, 119), ('2018-11-15', '2018-12-15', '2019-01-15', '2019-02-15', '2019-03-14') )
ax = sns.lineplot(data=data, linewidth=2.5)
ax.set_xlabel("time")
ax.set_ylabel("temperature")
plt.legend(loc = 'upper right')
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.show()