import pandas as pd
import zhplot
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 配置参数 ---
file_path = '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t1_待破壳.txt'  # 您的数据文件名
sampling_rate = 1000  # 信号的采样率 (Hz)
column_name_to_process = 'Signal_1' # 处理第一列

# 自定义滤波参数 (根据约5Hz心跳进行调整)
filter_lowcut = 1.0  # 低截止频率 (Hz)
filter_highcut = 40.0 # 高截止频率 (Hz)
filter_order = 2     # 滤波器阶数
powerline_freq = 50  # 工频干扰频率 (Hz)，根据地区调整

# R峰检测方法
peak_detection_method = "rodrigues2021"

# --- 2. 加载数据 ---
print(f"正在加载数据文件: {file_path}")
try:
    data_df = pd.read_csv(file_path, sep=r'\s+', skiprows=2, header=None, names=['Signal_1', 'Signal_2'])
    if column_name_to_process in data_df.columns:
        raw_signal = data_df[column_name_to_process].values
        print(f"成功加载并选择列 '{column_name_to_process}' 进行处理，共 {len(raw_signal)} 个数据点。")
    else:
        print(f"错误：在文件中未找到名为 '{column_name_to_process}' 的列。")
        exit()
except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。")
    exit()
except Exception as e:
    print(f"加载数据时出错：{e}")
    exit()

# --- 3. 自定义信号清洁 ---
print(f"\n正在使用自定义参数清洁信号 (Lowcut: {filter_lowcut} Hz, Highcut: {filter_highcut} Hz)...")
try:
    # 应用带通滤波器
    ecg_bandpassed = nk.signal_filter(raw_signal,
                                      sampling_rate=sampling_rate,
                                      lowcut=filter_lowcut,
                                      highcut=filter_highcut,
                                      method="butterworth",
                                      order=filter_order)
    print(f"带通滤波完成 (Butterworth, Order: {filter_order})。")

    # 应用工频干扰滤波器
    if powerline_freq > 0:
        ecg_cleaned = nk.signal_filter(ecg_bandpassed,
                                       sampling_rate=sampling_rate,
                                       method="powerline",
                                       powerline=powerline_freq)
        print(f"工频干扰滤波完成 (Frequency: {powerline_freq} Hz)。")
    else:
        ecg_cleaned = ecg_bandpassed
        print("跳过工频干扰滤波。")
        
except Exception as e:
    print(f"信号清洁过程中出错: {e}")
    print("将尝试使用原始信号进行R峰检测，但这可能会影响准确性。")
    ecg_cleaned = raw_signal

# --- 4. 使用 '{peak_detection_method}' 算法查找R峰 ---
print(f"\n正在使用 '{peak_detection_method}' 算法查找R峰...")
try:
    # nk.ecg_peaks 返回一个包含标记R峰的DataFrame (我们这里用 _ 忽略) 和一个包含R峰索引的info字典
    _, info = nk.ecg_peaks(ecg_cleaned, 
                           sampling_rate=sampling_rate, 
                           method=peak_detection_method, 
                           correct_artifacts=True) # 建议开启伪迹校正
    
    r_peaks_indices = info['ECG_R_Peaks'] # 从info字典中提取R峰的索引
    
    if len(r_peaks_indices) > 0:
        print(f"使用 '{peak_detection_method}' 算法成功检测到 {len(r_peaks_indices)} 个R峰。")

        # --- 4.1 计算心率 ---
        # 使用 nk.signal_rate 计算瞬时心率，然后计算平均值
        rate = nk.signal_rate(r_peaks_indices, #可以直接传入R峰索引
                              sampling_rate=sampling_rate,
                              desired_length=len(ecg_cleaned))
        average_heart_rate = np.nanmean(rate)
        print(f"平均心率: {average_heart_rate:.2f} bpm (次/分钟)")

    else:
        print(f"使用 '{peak_detection_method}' 算法未能检测到R峰。")
        average_heart_rate = None # 未检测到R峰，无法计算心率

except Exception as e:
    print(f"使用 '{peak_detection_method}' 算法进行R峰检测过程中出错: {e}")
    r_peaks_indices = [] # 如果检测失败，设置为空列表
    average_heart_rate = None

# --- 5. 标记R峰 (可视化) ---
print("\n正在生成R峰标记图...")
time_axis = np.arange(len(ecg_cleaned)) / sampling_rate
plt.figure(figsize=(15, 6))
plt.plot(time_axis, ecg_cleaned, label=f"清洁后信号 (BP: {filter_lowcut}-{filter_highcut}Hz)", color='blue', alpha=0.8)

if len(r_peaks_indices) > 0:
    plt.scatter(time_axis[r_peaks_indices], ecg_cleaned[r_peaks_indices], 
                color='red', s=60, label=f"R峰 (方法: {peak_detection_method})", zorder=5)
    title_text = f"MCG信号: R峰 ({peak_detection_method}) - {column_name_to_process}"
    if average_heart_rate is not None:
        title_text += f" - 平均心率: {average_heart_rate:.2f} BPM"
    plt.title(title_text, fontsize=14)
else:
    plt.title(f"MCG信号: 清洁后信号 ({column_name_to_process} - 未检测到R峰)", fontsize=14)

plt.xlabel("时间 (秒) / Time (s)", fontsize=12)
plt.ylabel("信号幅值 (pT)", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n处理和绘图完成。✅")