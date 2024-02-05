import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_csv('D:/HeatingOptimization/STCDRank/doc/excel/20220928/short_dingfu_low/excel/MAE_ablation.csv')
print(data)


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False 
#fontdict={"family": "KaiTi", "size": 15}


# 将linear结果除以2，以便于展示
print(data.loc[0, 'MAPE'])
#data.loc[0, 'MAPE'] = data.loc[0, 'MAPE'] - 2.5
#data.loc[1, 'MAPE'] = data.loc[1, 'MAPE'] - 2.5
plt.figure(figsize=(6, 36/8))
x = range(len(data['methods']))
#print(x)
width = 0.5
x_array = np.array(x)
#print(x_array)
#print(data['Heat Temp'])
plt.bar(x_array, data['MAPE'], width = width, alpha = 0.8)
#plt.bar(x_array, data['Outdoor Temp'], width = width, alpha = 0.8, label='ST of Outdoor Temp')
#plt.bar(x_array + width, data['Model'], width = width, alpha = 0.8, label='ST of Model')
plt.xlabel("Models", fontsize = 17)
plt.ylabel("Percentage", fontsize = 17)
plt.xticks(x, data['methods'])
#y_ticks = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.2, 3.0, 3.5]
#y_ticks_labels = [0, 0.3, 0.6, 0.9, 1.2, 1.5, '~', 6.5, 7.0]
#y_ticks = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5]
#y_ticks_labels = [0, 0.5, 1, 1.5, '~', 5, 5.5, 6.0]
#plt.yticks(y_ticks, y_ticks_labels)
#plt.title("Standard Trustworthiness", fontsize = 19)
plt.title("MAPE", fontsize = 19)
#plt.legend(fontsize = 15)
plt.tick_params(labelsize=13)
plt.xticks(rotation=-30)
#plt.legend(fontsize = 15)
#plt.ylim((0, 40))
plt.tight_layout()
plt.show()
#plt.savefig("vis.jpg")
