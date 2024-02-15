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

# 数据
## 图1
#data = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/long_predict_acc.csv')
data = pd.read_csv(path + '/long_predict_acc.csv')
print(data)

#data_2 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/long_predict_smrPD.csv')
data_2 = pd.read_csv(path + '/long_predict_smrPD.csv')
print(data_2)



#fig = plt.figure()
#创建图形
#fig = plt.figure(figsize=(18, 6))
fig = plt.figure(figsize=(16, 6))
## 设置比例
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
ax1_1 = fig.add_subplot(spec[0])
#ax1_1 = plt.subplot(1,3,1,figsize=(10, 6))

### 第一个图的第一条线
#选择ax1
plt.sca(ax1_1)
#fig, ax = plt.subplots(figsize=(8, 6))
ax1_1.plot(data['long_predict_len'].values, data['LSTM'].values, 'o-', color='#81D4FA', linewidth=3, label='MAPE of LSTM')
ax1_1.plot(data['long_predict_len'].values, data['STCD_LSTM_DF_TC_loss_SC'].values, 'o-', color='#0277BD', linewidth=3, label='MAPE of CONST-LSTM')
ax1_1.set_xlabel('Predict Window length', fontsize=23)
ax1_1.set_ylabel('MAPE', fontsize=23)
#plt.title('(a) Feature Window Length', fontsize=23)
ax1_1.set_title("(a) Long-term Prediction", fontsize = 28)

ax1_1.tick_params(labelsize=15)
#plt.xticks(rotation=-30)
plt.legend(fontsize = 17, loc='center left')

### 第一个图的第二条线
ax1_2 = ax1_1.twinx()

ax1_2.plot(data_2['long_predict_len'].values, data_2['LSTM'].values, 's-', color='#FFAB91', linewidth=3, label='cRPD of LSTM')
ax1_2.plot(data_2['long_predict_len'].values, data_2['STCD_LSTM_DF_TC_loss_SC'].values, 's-', color='#D84315', linewidth=3, label='cRPD of CONST-LSTM')

ax1_2.set_ylabel('cRPD', fontsize=23)

ax1_2.tick_params(labelsize=19)
plt.legend(fontsize = 17, loc='center right')

#plt.title('(a) Predict Window Length', fontsize=23)




## 第二个图
### 数据
#data_1 = pd.read_csv('D:/HeatingOptimization/STCD_202306/result/CONST_RPD/different_feature/different_feature.csv')
data_1 = pd.read_csv(path + '/different_feature_qintaoyuan_low.csv')

### 选择第二个图
ax2_1 = fig.add_subplot(spec[1])

## 第二个图第一条线
plt.sca(ax2_1)

ax2_1.plot(data_1['diff_heat'].values, data_1['Heat_temp'].values, 'o-', linewidth=3, label='Heat temperature')
ax2_1.plot(data_1['diff_heat'].values, data_1['Outdoor_temp'].values, 'o-', linewidth=3, label='Outdoor temperature')

ax2_1.set_xlabel('Temperature difference', fontsize=23)
ax2_1.set_ylabel('Indoor temperature difference', fontsize=23)
ax2_1.set_title("(b) RPD plots of different factors", fontsize = 28)


ax2_1.tick_params(labelsize=19)
plt.legend(fontsize = 17, loc='lower right')



plt.tight_layout()
plt.show()
