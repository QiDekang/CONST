import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}


data = pd.read_csv('D:/HeatingOptimization/STCDRank/doc/excel/20220928/short_dingfu_low/excel/original_trustworthiness_outdoor_temp.csv')
print(data)
plt.figure(figsize=(8, 6))
#plt.plot(data['heat_diff'], data['DNN'], 'o-', linewidth=3, label='DNN')
plt.plot(data['diff_heat'], data['Fitted_physical_model'], '-', linewidth=3, label='Fitted_physical_model')
plt.plot(data['diff_heat'], data['Linear'], '-', linewidth=3, label='Linear')
plt.plot(data['diff_heat'], data['DNN'], '-', linewidth=3, label='DNN')
plt.plot(data['diff_heat'], data['LSTM'], '-', linewidth=3, label='LSTM')
plt.plot(data['diff_heat'], data['ResNet'], '-', linewidth=3, label='ResNet')
plt.plot(data['diff_heat'], data['MDL'], '-', linewidth=3, label='MDL')
plt.plot(data['diff_heat'], data['STCD_DNN'], '-', linewidth=3, label='STCD_DNN')
plt.plot(data['diff_heat'], data['STCD_LSTM'], '-', linewidth=3, label='STCD_LSTM')
plt.plot(data['diff_heat'], data['STCD_ResNet'], '-', linewidth=3, label='STCD_ResNet')
plt.plot(data['diff_heat'], data['STCD_MDL'], '-', linewidth=3, label='STCD_MDL')
plt.xlabel("Heat Temperature Difference", fontsize = 17)
plt.ylabel("Indoor Temperature Difference", fontsize = 17)
plt.title("Original Trustworthiness of Outdoor Temperature", fontsize = 19)
plt.legend(fontsize = 13, loc = 'upper left')
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()
