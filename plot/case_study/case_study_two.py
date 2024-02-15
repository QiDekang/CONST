import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}

# 数据
## 图1
data = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/case_study/case_study_PD.csv')



#fig = plt.figure()
#创建图形
#fig = plt.figure(figsize=(18, 6))
fig = plt.figure(figsize=(16, 8))
## 设置比例
#spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[1, 0.8, 1.2])
spec = gridspec.GridSpec(ncols=2, nrows=1)
'''
#plt.figure()
#第一行第一列图形
ax1 = plt.subplot(1,3,1)
#第一行第二列图形
ax2 = plt.subplot(1,3,2)
#第一行第三列图形
ax3 = plt.subplot(1,3,3)
'''
## 第一个图
ax1 = fig.add_subplot(spec[0])
#ax1_1 = plt.subplot(1,3,1,figsize=(10, 6))


#plt.plot(data['heat_diff'], data['DNN'], 'o-', linewidth=3, label='DNN')
ax1.plot(data['heat'], data['Fitted_physical_model'], '-', linewidth=3, label='Physical')
ax1.plot(data['heat'], data['Linear'], '-', linewidth=3, label='Linear')
ax1.plot(data['heat'], data['DNN_multi_time'], '-', linewidth=3, label='MLP')
ax1.plot(data['heat'], data['ResNet'], '-', linewidth=3, label='ResNet')
ax1.plot(data['heat'], data['MDL_DNN'], '-', linewidth=3, label='MDL')
ax1.plot(data['heat'], data['LSTM'], '-', linewidth=3, label='LSTM')
ax1.plot(data['heat'], data['STCD_DNN_DF_TC_loss_SC'], '-', linewidth=3, label='STTPF_MLP')
ax1.plot(data['heat'], data['STCD_ResNet_all'], '-', linewidth=3, label='STTPF_ResNet')
ax1.plot(data['heat'], data['STCD_MDL_all'], '-', linewidth=3, label='STTPF_MDL')
#plt.plot(data['diff_heat'], data['STCD_LSTM_DF_TC_loss_SC'], '-', color='#45d9fd', linewidth=3, label='STTPF_LSTM')
ax1.plot(data['heat'], data['STCD_LSTM_DF_TC_loss_SC'], '-', linewidth=3, label='STTPF_LSTM')

#plt.axhline(y=0.1, color='r', linestyle='-')
plt.xlabel("Heat Temperature", fontsize = 22)
plt.ylabel("Indoor Temperature", fontsize = 22)
plt.title("(a) Partial Dependence Plot", fontsize = 24)
plt.legend(fontsize = 16, loc = 'upper left')
plt.tick_params(labelsize=20)







## 第三个图
## 第三个图
### 数据
data = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/case_study/case_study_rPD.csv')
### 选择第三个图
ax3 = fig.add_subplot(spec[1])
ax3.plot(data['diff_heat'], data['Fitted_physical_model'], '-', linewidth=3, label='Physical')
ax3.plot(data['diff_heat'], data['Linear'], '-', linewidth=3, label='Linear')
ax3.plot(data['diff_heat'], data['DNN_multi_time'], '-', linewidth=3, label='MLP')
ax3.plot(data['diff_heat'], data['ResNet'], '-', linewidth=3, label='ResNet')
ax3.plot(data['diff_heat'], data['MDL_DNN'], '-', linewidth=3, label='MDL')
ax3.plot(data['diff_heat'], data['LSTM'], '-', linewidth=3, label='LSTM')
ax3.plot(data['diff_heat'], data['STCD_DNN_DF_TC_loss_SC'], '-', linewidth=3, label='STTPF_MLP')
ax3.plot(data['diff_heat'], data['STCD_ResNet_all'], '-', linewidth=3, label='STTPF_ResNet')
ax3.plot(data['diff_heat'], data['STCD_MDL_all'], '-', linewidth=3, label='STTPF_MDL')
#plt.plot(data['diff_heat'], data['STCD_LSTM_DF_TC_loss_SC'], '-', color='#45d9fd', linewidth=3, label='STTPF_LSTM')
ax3.plot(data['diff_heat'], data['STCD_LSTM_DF_TC_loss_SC'], '-', linewidth=3, label='STTPF_LSTM')

ax3.axhline(y=0.1, color='r', linestyle='--')
#ax3.plot(data_2_1['models'], data_2_1['Acc'], 'o-', color='#47b8e0', linewidth=3, label='Acc')
ax3.set_xlabel('Heat Temperature Difference', fontsize=22)
ax3.set_ylabel('Indoor Temperature Difference', fontsize=22)
ax3.tick_params(labelsize=20)
plt.title('(b) Restricted Partial Dependence Plot', fontsize=24)
plt.legend(fontsize = 16, loc = 'lower right')
#plt.legend(fontsize = 15, loc='lower left')



plt.tight_layout()
plt.show()
