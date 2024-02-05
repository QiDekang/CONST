
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}


data = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/long_predict_acc.csv')
print(data)

data_2 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/long_predict_smrPD.csv')
print(data_2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data['long_predict_len'], data['LSTM'], 'o-', color='#81D4FA', linewidth=3, label='MAPE of LSTM')
ax.plot(data['long_predict_len'], data['STCD_LSTM_DF_TC_loss_SC'], 'o-', color='#0277BD', linewidth=3, label='MAPE of STTCD-LSTM')
ax.set_xlabel('Predict Window length', fontsize=19)
ax.set_ylabel('MAPE', fontsize=19)
#ax.tick_params(axis='y',labelsize=8)
ax.tick_params(labelsize=15)
#plt.xticks(rotation=-30)
plt.legend(fontsize = 15, loc='center left')

#plt.tight_layout()
#plt.show()


ax2 = ax.twinx()

ax2.plot(data_2['long_predict_len'], data_2['LSTM'], 's-', color='#FFAB91', linewidth=3, label='smrPD of LSTM')
ax2.plot(data_2['long_predict_len'], data_2['STCD_LSTM_DF_TC_loss_SC'], 's-', color='#D84315', linewidth=3, label='smrPD of STCD-LSTM')

ax2.set_ylabel('SMT', fontsize=19)
ax2.tick_params(labelsize=15)
#ax2.tick_params('y', colors='r')

#plt.legend(loc=1)
plt.legend(fontsize = 15, loc='center right')



#plt.xticks(rotation=-30)
#plt.title('Predict Window Length', fontsize=23)
plt.tight_layout()
plt.show()
