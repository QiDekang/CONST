
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}


data = pd.read_csv('D:/HeatingOptimization/STCD_2023/data/data_preparation/all_sensors_average/dingfu_high_data.csv')
#data = pd.read_csv('D:/HeatingOptimization/STCD_2023/data/data_preparation/all_sensors_average/qintaoyuan_low_data.csv')

print(data)
plt.figure(figsize=(8, 4.5))
'''
plt.plot(data['time'], data['indoor_temp'], '-', linewidth=3, label='indoor_temp')
plt.plot(data['time'], data['second_heat_temp'], '-', linewidth=3, label='second_heat_temp')
plt.plot(data['time'], data['outdoor_temp'], '-', linewidth=3, label='outdoor_temp')


plt.xlabel("Heat Temperature Difference", fontsize = 17)
plt.ylabel("Indoor Temperature Difference", fontsize = 17)
plt.title("Original Trustworthiness of Heat Temperature", fontsize = 19)
plt.legend(fontsize = 13, loc = 'upper left')
plt.tick_params(labelsize=15)
plt.tight_layout()
'''
ax = sns.lineplot(x="time", y="indoor_temp", data=data)
ax = sns.lineplot(x="time", y="second_heat_temp", data=data)
ax = sns.lineplot(x="time", y="outdoor_temp", data=data)
plt.show()
