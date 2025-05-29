import numpy as np
import zhplot
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks






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



"""定义绘图函数"""
def plot_signals_with_r_peaks(time, Bx_raw, Bx_filtered, By_raw, By_filtered, R_peaks_Bx, R_peaks_By):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True)

    # Bx 原始信号
    axs[0, 0].plot(time, Bx_raw, label='Bx_Raw', color='royalblue', alpha=0.7)
    axs[0, 0].set_title('Bx Raw Signal')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    plt.xlim(0, 40)
    axs[0, 0].set_ylim(0, 30)

    # By 原始信号
    axs[0, 1].plot(time, By_raw, label='By_Raw', color='royalblue', alpha=0.7)
    axs[0, 1].set_title('By Raw Signal')
    axs[0, 1].set_xlabel('Time(s)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    plt.xlim(0, 40)
    axs[0, 1].set_ylim(0, 30)

    # Bx 滤波信号及R峰
    axs[1, 0].plot(time, Bx_filtered, label='Bx_filtered', color='royalblue')
    if len(R_peaks_Bx) > 0:
        axs[1, 0].scatter(time[R_peaks_Bx], Bx_filtered[R_peaks_Bx], facecolors='none', edgecolors='r', marker='o', label='Bx_R peaks')
    axs[1, 0].set_title('Bx Filtered Signal with R Peaks')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    plt.xlim(0,40)
    axs[1, 0].set_ylim(-3, 3)

    # By 滤波信号及R峰
    axs[1, 1].plot(time, By_filtered, label='By_filtered', color='royalblue')
    if len(R_peaks_By) > 0:
        axs[1, 1].scatter(time[R_peaks_By], By_filtered[R_peaks_By], facecolors='none', edgecolors='r', marker='o', label='By_R peaks')
    axs[1, 1].set_title('By Filtered Signal with R Peaks')
    axs[1, 1].set_xlabel('Time(s)')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    plt.xlim(0,40)
    axs[1, 1].set_ylim(-3, 3)

    plt.tight_layout()
    plt.show()

    