import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
import os


#current_path = os.getcwd()
path = os.path.dirname(os.path.abspath(__file__))
print(path)

mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}




#fig = plt.figure()
#创建图形
#fig = plt.figure(figsize=(18, 6))
fig = plt.figure(figsize=(16, 9))
## 设置比例
spec = gridspec.GridSpec(ncols=2, nrows=2)
#spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[1, 1, 1])
'''
#plt.figure()
#第一行第一列图形
ax1 = plt.subplot(1,3,1)
#第一行第二列图形
ax2 = plt.subplot(1,3,2)
#第一行第三列图形
ax3 = plt.subplot(1,3,3)
'''

# 特征时间窗口
## 第二个图
### 数据
#data_1 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/feature_window_len/feature_window_len_Acc.csv')
data_1 = pd.read_csv(path + '/feature_window_len_Acc.csv')
#data_2 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/feature_window_len/feature_window_len_smrPD.csv')
data_2 = pd.read_csv(path + '/feature_window_len_smrPD.csv')
### 选择第二个图
ax2_1 = fig.add_subplot(spec[0])

## 第二个图第一条线
plt.sca(ax2_1)

ax2_1.plot(data_1['windows_len'].values, data_1['A'].values, 'o-', color='#81D4FA', linewidth=3, label='Acc-A')
ax2_1.plot(data_1['windows_len'].values, data_1['B'].values, 'o-', color='#B3E5FC', linewidth=3, label='Acc-B')
ax2_1.plot(data_1['windows_len'].values, data_1['C'].values, 'o-', color='#E1F5FE', linewidth=3, label='Acc-C')
ax2_1.plot(data_1['windows_len'].values, data_1['D'].values, 'o-', color='#01579B', linewidth=3, label='Acc-D')

ax2_1.set_xlabel('Feature Window length', fontsize=23)
ax2_1.set_ylabel('Accuracy', fontsize=23)
ax2_1.set_title("(a) The influence of feature window length", fontsize = 26)


ax2_1.tick_params(labelsize=17)
plt.legend(fontsize = 15, loc='lower left')


ax2_2 = ax2_1.twinx()

ax2_2.plot(data_2['windows_len'].values, data_2['A'].values, 's-', color='#FFAB91', linewidth=3, label='cRPD-A')
ax2_2.plot(data_2['windows_len'].values, data_2['B'].values, 's-', color='#FFCCBC', linewidth=3, label='cRPD-B')
ax2_2.plot(data_2['windows_len'].values, data_2['C'].values, 's-', color='#FBE9E7', linewidth=3, label='cRPD-C')
ax2_2.plot(data_2['windows_len'].values, data_2['D'].values, 's-', color='#BF360C', linewidth=3, label='cRPD-D')


ax2_2.set_ylabel('cRPD', fontsize=23)


ax2_2.tick_params(labelsize=17)
ax2_2.set_ylim((40, 100))
plt.legend(fontsize = 15, loc='lower right')
#plt.title('(b) Feature Window Length', fontsize=23)





# 数据
## 图1
#data = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/TC_loss_weights/TC_loss_weights_Acc.csv')
data = pd.read_csv(path + '/TC_loss_weights_Acc.csv')
print(data)

#data_2 = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/TC_loss_weights/TC_loss_weights_SMT.csv')
data_2 = pd.read_csv(path + '/TC_loss_weights_SMT.csv')
print(data_2)

## 第一个图
ax1_1 = fig.add_subplot(spec[1])
#ax1_1 = plt.subplot(1,3,1,figsize=(10, 6))

### 第一个图的第一条线
#选择ax1
plt.sca(ax1_1)
ax1_1.plot(data['TC_loss_weights'].values, data['Acc'].values, 'o-', color='#81D4FA', linewidth=3, label='Acc')
ax1_1.set_xlabel('TGT loss weight', fontsize=23)
ax1_1.set_ylabel('Accuracy', fontsize=23)

ax1_1.tick_params(labelsize=17)
#plt.xticks(rotation=-30)
plt.legend(fontsize = 19, loc='center left')

### 第一个图的第二条线
ax1_2 = ax1_1.twinx()

ax1_2.plot(data_2['TC_loss_weights'].values, data_2['SMT'].values, 's-', color='#FFAB91', linewidth=3, label='cRPD')

ax1_2.set_ylabel('cRPD', fontsize=23)
ax1_2.tick_params(labelsize=17)
plt.legend(fontsize = 19, loc='center right')

plt.title('(b) The influence of loss weight', fontsize=26)




## 第二个图
### 数据
#data_2_1 = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/periodic_effect_rate/periodic_effect_rate_Acc.csv')
data_2_1 = pd.read_csv(path + '/periodic_effect_rate_Acc.csv')
print(data_2_1)

#data_2_2 = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/periodic_effect_rate/periodic_effect_rate_SMT.csv')
data_2_2 = pd.read_csv(path + '/periodic_effect_rate_SMT.csv')
print(data_2_2)
### 选择第二个图
ax2_1 = fig.add_subplot(spec[3])

## 第二个图第一条线
plt.sca(ax2_1)

ax2_1.plot(data_2_1['periodic_effect_rate'].values, data_2_1['Acc'].values, 'o-', color='#81D4FA', linewidth=3, label='Acc')
ax2_1.set_xlabel('Periodic discount coefficient', fontsize=23)
ax2_1.set_ylabel('Accuracy', fontsize=23)

ax2_1.tick_params(labelsize=17)
plt.legend(fontsize = 19, loc='lower left')


ax2_2 = ax2_1.twinx()

ax2_2.plot(data_2_2['periodic_effect_rate'].values, data_2_2['SMT'].values, 's-', color='#FFAB91', linewidth=3, label='cRPD')
ax2_2.set_ylabel('cRPD', fontsize=23)
ax2_2.tick_params(labelsize=17)
plt.legend(fontsize = 19, loc='lower right')
plt.title('(d) The influence of temporal discount', fontsize=26)



## 第三个图
### 数据
#data_3_1 = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/enhancement_times/enhancement_times_Acc.csv')
data_3_1 = pd.read_csv(path + '/enhancement_times_Acc.csv')
print(data_3_1)

#data_3_2 = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/enhancement_times/enhancement_times_SMT.csv')
data_3_2 = pd.read_csv(path + '/enhancement_times_SMT.csv')
print(data_3_2)

### 选择第三个图
ax3_1 = fig.add_subplot(spec[2])
ax3_1.plot(data_3_1['enhancement_times'].values, data_3_1['Acc'].values, 'o-', color='#81D4FA', linewidth=3, label='Acc')
ax3_1.set_xlabel('Temporal enhancement times', fontsize=23)
ax3_1.set_ylabel('Accuracy', fontsize=23)
ax3_1.tick_params(labelsize=17)
plt.legend(fontsize = 19, loc='center left')

ax3_2 = ax3_1.twinx()

ax3_2.plot(data_3_2['enhancement_times'].values, data_3_2['SMT'].values, 's-', color='#FFAB91', linewidth=3, label='cRPD')

ax3_2.set_ylabel('cRPD', fontsize=23)
ax3_2.tick_params(labelsize=17)
#ax2.tick_params('y', colors='r')

#plt.legend(loc=1)
plt.legend(fontsize = 15, loc='center right')



plt.title('(c) The influence of data enhancement', fontsize=26)
#plt.legend(fontsize = 15, loc = 'upper left')
#plt.legend(fontsize = 15, loc='lower left')



plt.tight_layout()
plt.show()

