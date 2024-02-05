# 导入包
import numpy as np
import pandas as pd
import scipy as sc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math



def fitted_physical_model(data, params, k_return):
    
    '''
    w, k_1, b_1, k_2, b_2 = params
    middle_result = k_1 * ((44.5 ** (1+b))/25) * (1 - k_return) * data['second_heat_temp'].values + (44.5 ** (1+b)) * b_1
    result_1 = 0.5 * (1 + k_return) * data['second_heat_temp'].values - middle_result ** (1/(1+b))
    result_2 = k_2 * (1 - k_return) * data['second_heat_temp'].values + 25 * b_2 + data['outdoor_temp'].values
    model_result = w * result_1 + (1 - w) * result_2
    '''
    '''
    a_1, b_1, e_1, a_2, b_2, e_2, w, b, k_return = params
    #print(params)
    model_result = w*(((1+k_return)/2)*a_1*data['second_heat_temp'].values-b_1*(((44.5**(1+b)/25)*(1-k_return)*data['second_heat_temp'].values)**(1/(1+b)))+e_1)+(1-w)*(a_2*(1-k_return)*data['second_heat_temp'].values+b_2*data['outdoor_temp'].values+e_2)
    '''
    
    a_1, b_1, e_1, a_2, b_2, e_2, w, b = params
    middle_result = ((44.5 ** (1+b))/25) * (1 - k_return) * data['second_heat_temp'].values
    result_1 = 0.5 * (1 + k_return) * a_1 * data['second_heat_temp'].values - b_1 * (middle_result ** (1/(1+b))) + e_1
    result_2 = a_2 * (1 - k_return) * data['second_heat_temp'].values + b_2 * data['outdoor_temp'].values + e_2
    model_result = w * result_1 + (1 - w) * result_2

    return model_result
'''
def get_fitted_physical_model(data, name, is_fitted):
    
    if is_fitted == True:
        if name == 'dingfu_high':
            # [a_1, b_1, e_1, a_2, b_2, e_2, w, b]
            params = [-1.3387, -10.0000, 3.8850, -1.3465, 6.9023, 3.8860, 0.9805, 0.1600]
            # T(h) = k*T(g)
            k_return = 0.9286
        elif name == 'dingfu_low':
            params = [-2.2984, -9.7112, -6.5561, -2.2951, 3.7102, -6.5309, 0.9328, 0.1600]
            k_return = 0.8675
        elif name == 'qintaoyuan_high':
            params = [-3.0970, -10.0000, -6.0396, -3.0948, 6.2172, -6.0359, 0.9800, 0.1600]
            k_return = 0.8314
        elif name == 'qintaoyuan_low':
            # -10~10
            params = [-2.9496, -9.9997, -7.0433, -3.0403, 9.0227, -7.2230, 0.9896, 0.1600]
            k_return = 0.8387
    else:
        params = [1, 1, 0, 1, 1, 0, 0.5, 0.16]
        k_return = 0.8387

    model_result = fitted_physical_model(data, params, k_return)

    return model_result
'''
'''
def get_fitted_physical_model(data, name, is_fitted):
    
    # lb_ub_-5_5
    if is_fitted == True:
        if name == 'dingfu_high':
            # [a_1, b_1, e_1, a_2, b_2, e_2, w, b]
            params = [-0.4733, -5.0000, 5.0000, -0.4430, 5.0000, 5.0000, 0.9654, 0.1600]
            # T(h) = k*T(g)
            k_return = 0.9286
        elif name == 'dingfu_low':
            params = [-1.0271, -4.9991, 0.9929, -1.0282, 2.3663, 0.9936, 0.8955, 0.1600]
            k_return = 0.8675
        elif name == 'qintaoyuan_high':
            params = [-1.4003, -4.9995, 3.8420, -1.4011, 3.2046, 3.8414, 0.9610, 0.1600]
            k_return = 0.8314
        elif name == 'qintaoyuan_low':
            # -5~5
            params = [-1.3171, -5.0000, 2.6844, -1.3171, 2.5431, 2.6844, 0.9627, 0.1600]
            k_return = 0.8387
    else:
        params = [1, 1, 0, 1, 1, 0, 0.5, 0.16]
        k_return = 0.8387

    model_result = fitted_physical_model(data, params, k_return)

    return model_result
'''
'''
def get_fitted_physical_model(data, name, is_fitted):
    
    # lb_ub_-10_10
    if is_fitted == True:
        if name == 'dingfu_high':
            # [a_1, b_1, e_1, a_2, b_2, e_2, w, b]
            params = [-1.3387, -10.0000, 3.8850, -1.3465, 6.9023, 3.8860, 0.9805, 0.1600]
            # T(h) = k*T(g)
            k_return = 0.9286
        elif name == 'dingfu_low':
            params = [-2.2984, -9.7112, -6.5561, -2.2951, 3.7102, -6.5309, 0.9328, 0.1600]
            k_return = 0.8675
        elif name == 'qintaoyuan_high':
            params = [-3.0970, -10.0000, -6.0396, -3.0948, 6.2172, -6.0359, 0.9800, 0.1600]
            k_return = 0.8314
        elif name == 'qintaoyuan_low':
            params = [-2.9496, -9.9997, -7.0433, -3.0403, 9.0227, -7.2230, 0.9896, 0.1600]
            k_return = 0.8387
    else:
        params = [1, 1, 0, 1, 1, 0, 0.5, 0.16]
        k_return = 0.8387

    model_result = fitted_physical_model(data, params, k_return)

    return model_result
'''
'''
def get_fitted_physical_model(data, name, is_fitted):
    
    # lb_ub_-inf_inf
    if is_fitted == True:
        if name == 'dingfu_high':
            # [a_1, b_1, e_1, a_2, b_2, e_2, w, b]
            params = [-0.1075, -7.8998, -17.5476, 1.5897, 0.1658, 14.4593, 0.1040, 0.1600]
            # T(h) = k*T(g)
            k_return = 0.9286
        elif name == 'dingfu_low':
            params = [-0.3533, -5.0777, 15.7972, -3.1646, 0.4955, -8.0710, 0.4978, 0.1600]
            k_return = 0.8675
        elif name == 'qintaoyuan_high':
            params = [-1.7722, -7.8630, 47.0009, -1.7169, 0.2639, -37.2027, 0.5004, 0.1600]
            k_return = 0.8314
        elif name == 'qintaoyuan_low':
            params = [-87.9837, -69.4285, -40.1642, 393.7908, 0.1795, -72.5826, 0.5114, 0.1600]
            k_return = 0.8387
    else:
        params = [1, 1, 0, 1, 1, 0, 0.5, 0.16]
        k_return = 0.8387

    model_result = fitted_physical_model(data, params, k_return)

    return model_result
'''
def get_fitted_physical_model(data, name, is_fitted):
    
    # lb_ub_0_inf
    if is_fitted == True:
        if name == 'dingfu_high':
            # [a_1, b_1, e_1, a_2, b_2, e_2, w, b]
            params = [0.2703, 1.0562, 16.6199, 6.5484, 0.5514, 3.4227, 0.7565, 0.1600]
            # T(h) = k*T(g)
            k_return = 0.9286
        elif name == 'dingfu_low':
            params = [0.6782, 1.0134, 11.5625, 0.6766, 0.5614, 6.0775, 0.5646, 0.1600]
            k_return = 0.8675
        elif name == 'haihe_high':
            params = [0.4885, 1.3928, -20.8679, 3.7487, 0.7033, 587.1950, 0.9267, 0.1600]
            k_return = 0.8312
        elif name == 'qintaoyuan_high':
            params = [0.4479, 1.1676, 85.2156, 5.0703, 0.5127, -198.2745, 0.7526, 0.1600]
            k_return = 0.8314
        elif name == 'qintaoyuan_low':
            params = [0.4062, 1.0313, 34.9682, 4.6782, 0.3372, -35.2880, 0.6728, 0.1600]
            k_return = 0.8387
    else:
        params = [1, 1, 0, 1, 1, 0, 0.5, 0.16]
        k_return = 0.8387

    model_result = fitted_physical_model(data, params, k_return)

    return model_result

