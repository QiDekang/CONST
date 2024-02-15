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
data = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/long_predict_acc.csv')
print(data)

data_2 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/long_predict_smrPD.csv')
print(data_2)



#fig = plt.figure()
#创建图形
#fig = plt.figure(figsize=(18, 6))
fig = plt.figure(figsize=(16, 4.5))
## 设置比例
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[0.8, 1.2])
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
ax1_1 = fig.add_subplot(spec[0])
#ax1_1 = plt.subplot(1,3,1,figsize=(10, 6))

### 第一个图的第一条线
#选择ax1
plt.sca(ax1_1)
#fig, ax = plt.subplots(figsize=(8, 6))
ax1_1.plot(data['long_predict_len'], data['LSTM'], 'o-', color='#81D4FA', linewidth=3, label='MAPE of LSTM')
ax1_1.plot(data['long_predict_len'], data['STCD_LSTM_DF_TC_loss_SC'], 'o-', color='#0277BD', linewidth=3, label='MAPE of STTCD-LSTM')
ax1_1.set_xlabel('Predict Window length', fontsize=19)
ax1_1.set_ylabel('MAPE', fontsize=19)
#plt.title('(a) Feature Window Length', fontsize=23)
ax1_1.set_title("(a) Long-term Prediction", fontsize = 24)

ax1_1.tick_params(labelsize=15)
#plt.xticks(rotation=-30)
plt.legend(fontsize = 15, loc='lower left')

### 第一个图的第二条线
ax1_2 = ax1_1.twinx()

ax1_2.plot(data_2['long_predict_len'], data_2['LSTM'], 's-', color='#FFAB91', linewidth=3, label='smrPD of LSTM')
ax1_2.plot(data_2['long_predict_len'], data_2['STCD_LSTM_DF_TC_loss_SC'], 's-', color='#D84315', linewidth=3, label='smrPD of STCD-LSTM')

ax1_2.set_ylabel('smrPD', fontsize=19)

ax1_2.tick_params(labelsize=15)
plt.legend(fontsize = 15, loc='center right')

#plt.title('(a) Predict Window Length', fontsize=23)




## 第二个图
### 数据
data_1 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/feature_window_len/feature_window_len_MAPE.csv')

data_2 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/feature_window_len/feature_window_len_smrPD.csv')

### 选择第二个图
ax2_1 = fig.add_subplot(spec[1])

## 第二个图第一条线
plt.sca(ax2_1)

ax2_1.plot(data_1['windows_len'], data_1['A'], 'o-', color='#81D4FA', linewidth=3, label='A')
ax2_1.plot(data_1['windows_len'], data_1['B'], 'o-', color='#B3E5FC', linewidth=3, label='B')
ax2_1.plot(data_1['windows_len'], data_1['C'], 'o-', color='#E1F5FE', linewidth=3, label='C')
ax2_1.plot(data_1['windows_len'], data_1['D'], 'o-', color='#01579B', linewidth=3, label='D')

ax2_1.set_xlabel('Feature Window length', fontsize=19)
ax2_1.set_ylabel('MAPE', fontsize=19)
ax2_1.set_title("(b) Time Window Length of Feature", fontsize = 24)


ax2_1.tick_params(labelsize=15)
plt.legend(fontsize = 15, loc='lower left')


ax2_2 = ax2_1.twinx()

ax2_2.plot(data_2['windows_len'], data_2['A'], 's-', color='#FFAB91', linewidth=3, label='A')
ax2_2.plot(data_2['windows_len'], data_2['B'], 's-', color='#FFCCBC', linewidth=3, label='B')
ax2_2.plot(data_2['windows_len'], data_2['C'], 's-', color='#FBE9E7', linewidth=3, label='C')
ax2_2.plot(data_2['windows_len'], data_2['D'], 's-', color='#BF360C', linewidth=3, label='D')


ax2_2.set_ylabel('smrPD', fontsize=19)


ax2_2.tick_params(labelsize=15)
ax2_2.set_ylim((40, 100))
plt.legend(fontsize = 15, loc='lower right')
#plt.title('(b) Feature Window Length', fontsize=23)



plt.tight_layout()
plt.show()
