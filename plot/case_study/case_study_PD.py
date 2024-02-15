import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}


#data = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/OT/OT_qintaoyuan_low.csv')
data = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/case_study/case_study_PD.csv')
print(data)
plt.figure(figsize=(8, 4.5))
#plt.plot(data['heat_diff'], data['DNN'], 'o-', linewidth=3, label='DNN')
plt.plot(data['heat'], data['Fitted_physical_model'], '-', linewidth=3, label='Physical')
plt.plot(data['heat'], data['Linear'], '-', linewidth=3, label='Linear')
plt.plot(data['heat'], data['DNN_multi_time'], '-', linewidth=3, label='MLP')
plt.plot(data['heat'], data['ResNet'], '-', linewidth=3, label='ResNet')
plt.plot(data['heat'], data['MDL_DNN'], '-', linewidth=3, label='MDL')
plt.plot(data['heat'], data['LSTM'], '-', linewidth=3, label='LSTM')
plt.plot(data['heat'], data['STCD_DNN_DF_TC_loss_SC'], '-', linewidth=3, label='STTPF_MLP')
plt.plot(data['heat'], data['STCD_ResNet_all'], '-', linewidth=3, label='STTPF_ResNet')
plt.plot(data['heat'], data['STCD_MDL_all'], '-', linewidth=3, label='STTPF_MDL')
#plt.plot(data['diff_heat'], data['STCD_LSTM_DF_TC_loss_SC'], '-', color='#45d9fd', linewidth=3, label='STTPF_LSTM')
plt.plot(data['heat'], data['STCD_LSTM_DF_TC_loss_SC'], '-', linewidth=3, label='STTPF_LSTM')

#plt.axhline(y=0.1, color='r', linestyle='-')
plt.xlabel("Heat Temp", fontsize = 17)
plt.ylabel("Indoor Temp", fontsize = 17)
#plt.title("Original Trustworthiness of Heat Temperature", fontsize = 19)
plt.legend(fontsize = 13, loc = 'upper left')
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()
