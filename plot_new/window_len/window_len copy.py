import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}


data_1 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/feature_window_len/feature_window_len_MAPE.csv')
print(data_1)

data_2 = pd.read_csv('D:/HeatingOptimization/STCD_2023/result/cloud_all_results/window_len/feature_window_len/feature_window_len_smrPD.csv')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(data_1['windows_len'], data_1['A'], 'o-', color='#01579B', linewidth=2, label='A')
ax.plot(data_1['windows_len'], data_1['B'], 'o-', color='#B3E5FC', linewidth=2, label='B')
ax.plot(data_1['windows_len'], data_1['C'], 'o-', color='#E1F5FE', linewidth=3, label='C')
ax.plot(data_1['windows_len'], data_1['D'], 'o-', color='#81D4FA', linewidth=2, label='D')

ax.set_xlabel('Feature Window length', fontsize=19)
ax.set_ylabel('MAPE', fontsize=19)
#ax.tick_params(axis='y',labelsize=8)
ax.tick_params(labelsize=15)
#plt.xticks(rotation=-30)
plt.legend(fontsize = 15, loc='lower left')

#plt.tight_layout()
#plt.show()


ax2 = ax.twinx()

ax2.plot(data_2['windows_len'], data_2['A'], 's-', color='#BF360C', linewidth=2, label='A')
ax2.plot(data_2['windows_len'], data_2['B'], 's-', color='#FFCCBC', linewidth=2, label='B')
ax2.plot(data_2['windows_len'], data_2['C'], 's-', color='#FBE9E7', linewidth=3, label='C')
ax2.plot(data_2['windows_len'], data_2['D'], 's-', color='#FFAB91', linewidth=2, label='D')


ax2.set_ylabel('smrPD', fontsize=19)
ax2.tick_params(labelsize=15)
ax2.set_ylim((40, 100))
#ax2.tick_params('y', colors='r')

#plt.legend(loc=1)
plt.legend(fontsize = 15, loc='lower right')



#plt.xticks(rotation=-30)
#plt.title('Feature Window Length', fontsize=23)
plt.tight_layout()
plt.show()

'''
plt.figure(figsize=(8, 6))
#plt.plot(data['heat_diff'], data['DNN'], 'o-', linewidth=3, label='DNN')
plt.plot(data['models'], data['Acc'], 'o-', linewidth=3, label='Acc')
plt.xlabel("models", fontsize = 17)
plt.ylabel("Acc", fontsize = 17)
y_ticks = [96.5, 97, 97.5, 98, 98.5, 99, 99.5, 100]
y_ticks_labels = [93, 93.5, 94, 94.5, '~', 99, 99.5, 100]
plt.yticks(y_ticks, y_ticks_labels)
plt.title("ACC", fontsize = 19)
plt.legend(fontsize = 13, loc = 'upper left')
plt.xticks(rotation=-30)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()
'''
'''
        best
        upper right
        upper left
        lower left
        lower right
        right
        center left
        center right
        lower center
        upper center
        center
'''