
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks

"""Fig2 x,y轴原始信号、滤波信号及R峰展示图"""

# 1. 加载数据
data = np.loadtxt(r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t1_待破壳.txt', skiprows=2, encoding="utf-8")
Bx = data[:, 0]
By = data[:, 1]
fs = 1000  # 采样率

# 2. 检测参数设置

""" 设置滤波器参数 """
filter_order_bandpass = 4  # 带通滤波器的阶数 (根据用户最新提供)
lowcut_freq = 0.5          # Hz, 低截止频率
highcut_freq = 45.0        # Hz, 高截止频率
notch_freq_mains = 50.0    # Hz, 工频干扰频率
Q_factor_notch = 30.0      # 陷波滤波器的品质因数

""" 设置R峰检测参数 """
R_peak_min_height_factor = 0.4 
R_peak_min_distance_ms = 150   



# 定义函数

"""巴特沃斯滤波器"""
def bandpass_filter(data, fs, lowcut, highcut, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0: high = 0.99
    if low <= 0: low = 0.001
    if low >= high: return data
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfiltfilt(sos, data)

"""陷波滤波器"""
