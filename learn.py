import numpy as np
import zhplot
import os
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
def find_r_peaks_data(data, fs, min_height_factor, min_distance_ms, identifier="信号",percentile = 95):
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
## 第一张图：原始信号和滤波信号（包含R峰）
def plot_signals_with_r_peaks(time, Bx_raw, Bx_filtered, By_raw, By_filtered, R_peaks_Bx, R_peaks_By):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True)

    # Bx 原始信号
    axs[0, 0].plot(time, Bx_raw, label='Bx_Raw', color='royalblue', alpha=0.7)
    axs[0, 0].set_title('Bx Raw Signal')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_xlim(0, 40)

    # By 原始信号
    axs[0, 1].plot(time, By_raw, label='By_Raw', color='royalblue', alpha=0.7)
    axs[0, 1].set_title('By Raw Signal')
    axs[0, 1].set_xlabel('Time(s)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[0, 1].set_xlim(0, 40)

    # Bx 滤波信号及R峰
    axs[1, 0].plot(time, Bx_filtered, label='Bx_filtered', color='royalblue')
    if len(R_peaks_Bx) > 0:
        axs[1, 0].scatter(time[R_peaks_Bx], Bx_filtered[R_peaks_Bx], facecolors='none', edgecolors='r', marker='o', label='Bx_R peaks')
    axs[1, 0].set_title('Bx Filtered Signal with R Peaks')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[0, 0].set_xlim(0, 40)
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
    axs[1, 1].set_xlim(0, 40)
    axs[1, 1].set_ylim(-3, 3)

    plt.tight_layout()
    plt.show()
    return fig  # 返回图形对象以便后续处理或保存



### 求出平均心跳周期，绘图展示并保存
    
def averaged_cardias_cycle_plot(data, r_peaks_indices, fs,
                                pre_r_ms, post_r_ms, output_dir,base_filename,
                                identifier="信号"):
    if data is None or r_peaks_indices is None or len(r_peaks_indices) < 2:
        return False         # 如果数据或R峰索引无效，直接返回

    pre_r_samples = int(pre_r_ms / 1000 * fs)
    post_r_samples = int(post_r_ms / 1000 * fs)
    total_cycle_samples = pre_r_samples + post_r_samples

    if total_cycle_samples <= 0:
        print(f"  错误 ({identifier}): pre_r_ms 和 post_r_ms 的总和必须大于0。")
        return False          
    
    all_cycles = []
    valid_cycles_count = 0
    for r_peak in r_peaks_indices:
        start_index = r_peak - pre_r_samples
        end_index = r_peak + post_r_samples
        if start_index >= 0 and end_index <= len(data):  # 确保提取范围不超出原始数据边界
            cycle = data[start_index:end_index]
            all_cycles.append(cycle)
            valid_cycles_count += 1

    if valid_cycles_count < 2:
        print(f"  错误 ({identifier}): 有效的心跳周期少于2个，无法计算平均周期。")
        return False
    
    cycles_array = np.array(all_cycles)  # 将周期列表转换为NumPy数组


    """ --- R峰基线校正--- """
    # 1.计算R峰的原始平均周期和标准差
    averaged_cycle = np.mean(cycles_array, axis=0)   # 提取的所有周期的平均值
    std_cycle = np.std(cycles_array, axis=0)       # 提取的所有周期的标准差

    # 2.从原始平均周期中确定基线偏移量
    """ 使用 pre_r_ms 的前30ms作为基线
        窗口，如果 pre_r_ms 不足30ms，
        则使用整个 pre_r_ms """
    baseline_window_samples = int(min(pre_r_ms, 30) / 1000 * fs)  # 基线窗口大小
    if baseline_window_samples > 0 and baseline_window_samples <= pre_r_samples:
        baseline_offset = np.mean(averaged_cycle[:baseline_window_samples])
    else:
        baseline_offset = 0
        print(f"  错误 ({identifier}): 基线校正窗口无效")
    
    # 3.从平均周期中减去基线偏移量
    averaged_cycle_corrected = averaged_cycle - baseline_offset

    # 4.校正背景中的单个R峰周期
    background_cycles_corrected = []
    for cycle in cycles_array:
        if baseline_window_samples > 0 and baseline_window_samples <= pre_r_samples:
            individual_baseline_offset = np.mean(cycle[:baseline_window_samples])
            background_cycles_corrected.append(cycle - individual_baseline_offset)
        else:
            background_cycles_corrected.append(cycle)   # 如果窗口无效，不校正单个周期
    
    background_cycles_corrected = np.array(background_cycles_corrected)

    """ 基线校正完成 """

    """  绘制平均心跳周期 """
    # 1.设置x轴
    cycle_time_axis_r = np.linspace(-pre_r_ms / 1000, (post_r_ms-1) / 1000, total_cycle_samples)

    # 2.绘制背景周期数据（随机20条）
    fig, ax = plt.subplots(figsize = (10, 6))

    ax.set_title(f'{identifier} Averaged Cardiac Cycle\n(Based on {valid_cycles_count} cycles)', fontsize=16)
    ax.set_xlabel('Time relative to R-peak (s)', fontsize=12)
    ax.set_ylabel('Signal Magnitude (pT)', fontsize=12)
    
    num_bg_cycles_to_plot = min(len(background_cycles_corrected),20)
    
    if len(background_cycles_corrected) > num_bg_cycles_to_plot:
        indices_to_plot = np.random.choice(len(background_cycles_corrected), num_bg_cycles_to_plot, replace=False)
    else:
        indices_to_plot = np.arange(len(background_cycles_corrected))
    for i in indices_to_plot:
        ax.plot(cycle_time_axis_r, background_cycles_corrected[i,:], color='lightgray', alpha=0.35, linewidth=0.5)
    
    # 3.绘制校正后的平均心跳周期
    ax.plot(cycle_time_axis_r, averaged_cycle_corrected, color='royalblue', linewidth=2, label='Averaged Cycle')

    """  绘制标准差区域  """
    ax.fill_between(cycle_time_axis_r, 
                    averaged_cycle_corrected - std_cycle, 
                    averaged_cycle_corrected + std_cycle, 
                    color='royalblue', alpha=0.2, label='±1 Std Dev')
    
    ax.legend(loc='best')
    plt.tight_layout()

    try:
        output_image_filename = f"{base_filename}_{identifier}_averaged_cycle_corrected.png" ## 输出图片名
        output_dir = os.path.join(output_dir, output_image_filename)
    
        plt.savefig(output_dir, dpi=300, bbox_inches='tight')
        print(f"成功: 平均心跳周期图已保存到 {output_dir}")
        plt.show()  # 显示图形
    except Exception as e:
        print(f"错误: 保存平均心跳周期图时出错: {e}")
        return False
    finally:
        plt.close(fig)  # 关闭图形以释放内存
    

"""输入输出目录"""
input_dir = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t1_待破壳.txt'  # 输入数据文件路径
output_dir = r"C:\Users\Xiaoning Tan\Desktop\egg_figure"  # 输出目录



### ---- 主程序开始 ---- ###

# 1. 加载数据
data = np.loadtxt(input_dir, skiprows=2, encoding="utf-8")
Bx_raw = data[:, 0]
By_raw = data[:, 1]
fs = 1000  # 采样率

# 2. 检测参数设置

"""时间参数"""
time = np.arange(len(Bx_raw)) / fs  # 时间向量

""" 设置滤波器参数 """
filter_order_bandpass = 4  # 带通滤波器的阶数 (根据用户最新提供)
lowcut_freq = 0.5          # Hz, 低截止频率
highcut_freq = 45.0        # Hz, 高截止频率
notch_freq = 50.0    # Hz, 工频干扰频率
Q_factor_notch = 30.0      # 陷波滤波器的品质因数

""" 设置R峰检测参数 """
R_peak_min_height_factor = 0.6  # R峰最小高度因子 (相对于数据的最大值) 
R_peak_min_distance_ms = 200     # R峰最小距离 (毫秒)


""" 设置平均心跳周期参数 """
pre_r_ms = 100   # R峰前的时间窗口 (毫秒)
post_r_ms = 100  # R峰后的时间窗口 (毫秒)


# 自动从数据文件路径提取 base_filename
base_filename = os.path.splitext(os.path.basename(input_dir))[0]
output_dir = output_dir  # 保持后续变量一致


"""设置信号反转"""
reverse_Bx_signal = False  
if reverse_Bx_signal:
    Bx_raw = -Bx_raw  # 反转Bx信号
    print("信息: Bx 信号已反转。")

reverse_By_signal = False  
if reverse_By_signal:
    By_raw = -By_raw  # 反转By信号
    print("信息: By 信号已反转。")


# 3. 对原始数据进行滤波处理
print("开始滤波 Bx_raw 信号...")
Bx_filtered = bandpass_filter(Bx_raw, fs, lowcut_freq, highcut_freq, filter_order_bandpass)
print("开始滤波 By_raw 信号...")
By_filtered = bandpass_filter(By_raw, fs, lowcut_freq, highcut_freq, filter_order_bandpass)
print("滤波完成。")


## # 3.1 应用陷波滤波器去除工频干扰
Bx_filtered = apply_notch_filter(Bx_filtered, notch_freq, Q_factor_notch, fs)
By_filtered = apply_notch_filter(By_filtered, notch_freq, Q_factor_notch, fs)

# 4.寻找R峰
print("开始在Bx中寻找 R 峰...")
R_peaks_Bx = find_r_peaks_data(Bx_filtered, fs, R_peak_min_height_factor, R_peak_min_distance_ms, identifier="Bx信号")
print(f"在 Bx_filtered 中找到 {len(R_peaks_Bx)} 个R峰。")

print("开始在By中寻找 R 峰...")
R_peaks_By = find_r_peaks_data(By_filtered, fs, R_peak_min_height_factor, R_peak_min_distance_ms, identifier="By信号")
print(f"在 By_filtered 中找到 {len(R_peaks_By)} 个R峰。")

### 4.1 标记R峰为红色空心圆圈
R_peaks_Bx_y = Bx_filtered[R_peaks_Bx] if len(R_peaks_Bx) > 0 else np.array([])

### 调整Bx，By信号的y轴所在区间
Bx_raw += 5
By_raw += 8
Bx_filtered -= 0.5

# 5. 绘制结果
# 绘制Bx和By原始信号和滤波信号
print("开始绘制原始信号与滤波信号对比图...")
fig1 = plot_signals_with_r_peaks(time, Bx_raw, Bx_filtered, By_raw, By_filtered, R_peaks_Bx, R_peaks_By)
plt.close(fig1)  # 关闭图形以释放内存

# 6.绘制平均心跳周期
print("\n开始处理Bx信号的平均心跳周期...")
averaged_cardias_cycle_plot(
    data=Bx_filtered,
    r_peaks_indices=R_peaks_Bx,
    fs=fs,
    pre_r_ms=pre_r_ms,
    post_r_ms=post_r_ms,
    output_dir=output_dir, 
    base_filename=base_filename, 
    identifier="Bx_Filtered"
)
print("\n开始处理By信号的平均心跳周期...")
averaged_cardias_cycle_plot(
    data=By_filtered,
    r_peaks_indices=R_peaks_By,
    fs=fs,
    pre_r_ms=pre_r_ms,
    post_r_ms=post_r_ms,
    output_dir=output_dir, 
    base_filename=base_filename, 
    identifier="By_Filtered"
)

print("\n进程结束！！！")
