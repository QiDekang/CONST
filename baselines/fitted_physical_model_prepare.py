# 导入包
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


# 线性回归T(h) = k*T(g)+b
# 线性回归T(h) = k1*T(g) + k2*T(w) +b
def fitted_physical_model_prepare(data, file_name):

    #print(file_name)
    x_train = data['second_heat_temp'].values.reshape(-1, 1)
    #x_train = data[['second_heat_temp', 'outdoor_temp']].values
    y_train = data['second_return_temp'].values

    '''
    sns.set(style="ticks")
    plt.figure(figsize=(16,9))
    sns.set_context('notebook', font_scale=1.5)
    ax = sns.scatterplot(x=data['second_heat_temp'].values, y=data['second_return_temp'].values)
    ax.set_xlabel("Heat Temperature", fontsize = 28)
    ax.set_ylabel("Return Heat Temperature", fontsize = 28)
    plt.tick_params(labelsize=24)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.show()
    '''

    # 无截距，可避免出现t_g-t_h<0的情况。
    model = LinearRegression(fit_intercept=False)
    #model = LinearRegression()
    performance = model.fit(x_train, y_train)

    r2 = performance.score(x_train, y_train)
    #print('r2:', r2)

    k = round(model.coef_[0], 4)
    b = round(model.intercept_, 4)
    #print('k:', k)
    #print('b:', b)
    #print(model)

    #r2: 0.9557841884606767, 0.9517670155566974, 0.732411076666661, 0.7600949504665123.
    #k: 0.9286, 0.8675, 0.8314, 0.8387


    return model
