
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
notch_freq = 50.0    # Hz, 工频干扰频率
Q_factor_notch = 30.0      # 陷波滤波器的品质因数

""" 设置R峰检测参数 """
R_peak_min_height_factor = 0.45  # R峰最小高度因子 (相对于数据的最大值) 
R_peak_min_distance_ms = 200     # R峰最小距离 (毫秒)


"""设置信号反转"""
reverse_signal = False  # 是否反转信号
if reverse_signal:
    Bx = -Bx  # 反转Bx信号
    By = -By  # 反转By信号


# 3. 对原始数据进行滤波处理
Bx_filtered = bandpass_filter(Bx, fs, lowcut_freq, highcut_freq, filter_order_bandpass)
By_filtered = bandpass_filter(Bx, fs, lowcut_freq, highcut_freq, filter_order_bandpass)

## # 3.1 应用陷波滤波器去除工频干扰
Bx_filtered = apply_notch_filter(Bx_filtered, notch_freq, Q_factor_notch, fs)
By_filtered = apply_notch_filter(By_filtered, notch_freq, Q_factor_notch, fs)

# 4.寻找R峰
R_peaks_Bx = find_r_peaks_data(Bx_filtered, fs, R_peak_min_height_factor, R_peak_min_distance_ms, identifier="Bx信号")
R_peaks_By = find_r_peaks_data(By_filtered, fs, R_peak_min_height_factor, R_peak_min_distance_ms, identifier="By信号")


# 5. 绘制结果
"""5.1 绘制原始信号和滤波信号"""
plt.figure(figsize=(12, 8))




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
def apply_notch_filter(data, notch_freq, quality_factor, fs):
    if data is None: return None
    nyquist = 0.5 * fs
    freq_normalized = notch_freq / nyquist
    if freq_normalized >= 1.0 or freq_normalized <= 0: return data
    b, a = iirnotch(freq_normalized, quality_factor)
    return filtfilt(b, a, data)


"""R峰检测函数"""
def find_r_peaks_data(data, fs, min_height_factor, min_distance_ms, identifier="信号",percentile = 99):
    if data is None or len(data) == 0: return np.array([])
    data_max = np.percentile(data, percentile)
    if data_max <= 1e-9: return np.array([])  
    min_h = min_height_factor * data_max
    min_distance = int(min_distance_ms / 1000 * fs)

    try:
        peaks, _ = find_peaks(data, height=min_h, distance=min_distance)
    except Exception as e:
        print(f"  错误 ({identifier}): 调用 find_peaks 时出错: {e}")
        return np.array([])
    return peaks


