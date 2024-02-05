import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_csv('D:/HeatingOptimization/STCDRank/doc/excel/20220928/short_dingfu_low/excel/standard_trustworthiness_ablation.csv')
print(data)


mpl.rcParams['font.family'] = 'Times New Roman'
#mpl.rcParams['font.size'] = '15'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#fontdict={"family": "KaiTi", "size": 15}


plt.figure(figsize=(6, 36/8))
x = range(len(data['model']))
print(x)
width = 0.25
x_array = np.array(x)
print(x_array)
print(data['Heat Temp'])
plt.bar(x_array - width, data['Heat Temp'], width = width, alpha = 0.8, label='ST of Heat Temp')
plt.bar(x_array, data['Outdoor Temp'], width = width, alpha = 0.8, label='ST of Outdoor Temp')
plt.bar(x_array + width, data['Model'], width = width, alpha = 0.8, label='ST of Model')
#plt.axhline(y=100, color = 'red', linestyle = "--")
#plt.plot(x, y_1, 'o-', linewidth=3, label='活动力度')
#plt.plot(x, y_2, 'o-', linewidth=3, label='拉动率')
plt.xlabel("Models", fontsize = 17)
plt.ylabel("Percentage", fontsize = 17)
plt.xticks(x, data['model'])
plt.xticks(rotation=-30)
plt.title("Standard Trustworthiness", fontsize = 19)
#plt.legend(fontsize = 15)
plt.tick_params(labelsize=13)
plt.legend(fontsize = 15)
#plt.ylim((0, 40))
plt.tight_layout()
plt.show()
#plt.savefig("vis.jpg")
