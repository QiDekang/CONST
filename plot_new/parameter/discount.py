
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}


data = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/periodic_effect_rate/periodic_effect_rate_Acc.csv')
print(data)

data_2 = pd.read_csv('D:/HeatingOptimization/STCDRank/result/result_20221111/4_stations_physical_as_BT/periodic_effect_rate/periodic_effect_rate_SMT.csv')
print(data_2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data['periodic_effect_rate'], data['Acc'], 'o-', color='#81D4FA', linewidth=3, label='Acc')
ax.set_xlabel('Period discount factor', fontsize=19)
ax.set_ylabel('Acc', fontsize=19)
#ax.tick_params(axis='y',labelsize=8)
ax.tick_params(labelsize=15)
#plt.xticks(rotation=-30)
plt.legend(fontsize = 15, loc='lower left')

#plt.tight_layout()
#plt.show()


ax2 = ax.twinx()

ax2.plot(data_2['periodic_effect_rate'], data_2['SMT'], 's-', color='#FFAB91', linewidth=3, label='SMT')

ax2.set_ylabel('SMT', fontsize=19)
ax2.tick_params(labelsize=15)
#ax2.tick_params('y', colors='r')

#plt.legend(loc=1)
plt.legend(fontsize = 15, loc='center right')



#plt.xticks(rotation=-30)
plt.title('Period discount', fontsize=23)
plt.tight_layout()
plt.show()
