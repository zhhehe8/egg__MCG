import zhplot 
import matplotlib
# 尝试使用Qt5Agg后端以获得更好的交互性，如果不行可尝试 'TkAgg'
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import os # 导入os模块用于路径操作

# --- 配置参数 (用户可根据需要调整) ---
SAMPLING_RATE = 1000  # Hz, 从数据文件得知是1K
FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t3.txt' # 你的数据文件路径
SKIP_HEADER_LINES = 3 # 数据文件头部需要跳过的行数
FILE_ENCODING = 'utf-8' # 数据文件编码，通常是 'utf-8' 或 'gbk'
OUTPUT_FIGURE_DIR = r'C:\Users\Xiaoning Tan\Desktop\egg_figure' # *** 图片保存目录 ***

# 滤波参数 (通用)
FILTER_ORDER_BANDPASS = 4 # 带通滤波器的阶数 (建议3或4, 较低阶数可能减少失真)
LOWCUT_FREQ = 0.5         # Hz, 低截止频率 (去除基线漂移)
HIGHCUT_FREQ = 45.0       # Hz, 高截止频率 (尝试60-80Hz以观察Q波, 原为45Hz)
NOTCH_FREQ_MAINS = 50.0   # Hz, 工频干扰频率
Q_FACTOR_NOTCH = 30.0     # 陷波滤波器的品质因数 (原为30, 降低可使陷波略宽)

# ----------------------------------------------------------
# R峰检测参数 - 分通道设置
# ----------------------------------------------------------
# Bx 参数
R_PEAK_MIN_HEIGHT_FACTOR_CH1 = 0.25 # R峰检测的最小高度因子 (Bx, 范围建议 0.3-0.7)
R_PEAK_MIN_DISTANCE_MS_CH1 = 150   # R峰之间的最小距离 (毫秒, Bx)

# By 参数 (初始值可与Bx相同，根据需要独立调整)
R_PEAK_MIN_HEIGHT_FACTOR_CH2 = 0.4 # R峰检测的最小高度因子 (By, 范围建议 0.3-0.7)
R_PEAK_MIN_DISTANCE_MS_CH2 = 150   # R峰之间的最小距离 (毫秒, By)
# ----------------------------------------------------------

# 周期提取和平均参数 (通用)
PRE_R_PEAK_MS = 100       # ms, 提取周期时R峰前的时间
POST_R_PEAK_MS = 150      # ms, 提取周期时R峰后的时间 (用户设定的值)
NUM_INDIVIDUAL_CYCLES_TO_PLOT = 5 # 在叠加图上绘制多少个单独的周期

# --- 函数定义 ---

def load_cardiac_data(filepath, skip_header, file_encoding='utf-8'):
    """
    从文本文件加载胚鸡心磁数据。
    """
    try:
        data = np.loadtxt(filepath, skiprows=skip_header, dtype=float, encoding=file_encoding)
        if data.ndim == 1: # 处理单列数据的情况（如果可能）
            print(f"警告: 文件 {filepath} 只包含一列数据。将仅处理这一列。")
            return data, None # 返回None作为第二个通道
        if data.shape[1] < 2:
            print(f"错误：数据文件 {filepath} 至少需要两列才能处理两个通道。当前列数: {data.shape[1]}")
            return None, None
        channel1 = data[:, 0]
        channel2 = data[:, 1]
        return channel1, channel2
    except UnicodeDecodeError as ude:
        print(f"使用 '{file_encoding}' 编码读取文件 '{filepath}' 时出错: {ude}")
        print(f"如果编码不正确，请尝试其他编码，例如 'gbk' 或 'latin-1'。")
        return None, None
    except Exception as e:
        print(f"读取文件 '{filepath}' 或解析数据时出错: {e}")
        return None, None

def apply_bandpass_filter(data, lowcut, highcut, fs, order):
    """
    对数据应用带通巴特沃斯滤波器。
    """
    if data is None: return None
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0: # 确保高截止频率不超过奈奎斯特频率
        high = 0.99 # 或者给出错误/警告
        print(f"警告: highcut频率({highcut}Hz)过高,已调整为奈奎斯特频率的99%。")
    if low <= 0: # 低截止频率不能为0或负
        low = 0.001 # 或者给出错误/警告
        print(f"警告: lowcut频率({lowcut}Hz)过低,已调整为0.001*奈奎斯特频率。")
    if low >= high:
        print(f"错误: 带通滤波器的低截止({lowcut}Hz)必须小于高截止({highcut}Hz)。跳过带通滤波。")
        return data # 或者返回 None 或原始数据

    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def apply_notch_filter(data, notch_freq, quality_factor, fs):
    """
    对数据应用陷波滤波器以去除特定频率的噪声。
    """
    if data is None: return None
    nyquist = 0.5 * fs
    freq_normalized = notch_freq / nyquist
    if freq_normalized >= 1.0 or freq_normalized <= 0:
        print(f"警告: 陷波频率 {notch_freq}Hz (归一化后 {freq_normalized}) 无效。跳过陷波滤波。")
        return data
    b, a = iirnotch(freq_normalized, quality_factor)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def plot_signals_comparison(time_axis, original_data, bandpass_filtered_data, fully_filtered_data, channel_name,
                            output_dir=None, base_filename=""):
    """
    绘制原始信号、带通滤波后信号和完全滤波后信号的对比图，并保存。
    """
    if original_data is None:
        print(f"警告 ({channel_name}): 原始数据为空，无法绘制信号对比图。")
        return

    fig = plt.figure(figsize=(18, 10)) # 获取figure对象
    plt.suptitle(f'{channel_name} - 信号处理步骤对比', fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(time_axis, original_data, label='原始信号')
    plt.title('步骤 1: 原始信号')
    plt.xlabel('时间 (秒)')
    plt.ylabel('磁场强度 (pT)')
    plt.legend()
    plt.grid(True)

    if bandpass_filtered_data is not None:
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, bandpass_filtered_data, label=f'带通滤波 ({LOWCUT_FREQ}-{HIGHCUT_FREQ} Hz, {FILTER_ORDER_BANDPASS}阶)')
        plt.title('步骤 2: 带通滤波后')
        plt.xlabel('时间 (秒)')
        plt.ylabel('磁场强度 (pT)')
        plt.legend()
        plt.grid(True)
    else:
        print(f"警告 ({channel_name}): 带通滤波数据为空。")

    if fully_filtered_data is not None:
        active_filters = f'带通 + {NOTCH_FREQ_MAINS} Hz 陷波滤波' if bandpass_filtered_data is not None else f'{NOTCH_FREQ_MAINS} Hz 陷波滤波 (若带通失败)'
        plt.subplot(3, 1, 3)
        plt.plot(time_axis, fully_filtered_data, label=active_filters)
        plt.title('步骤 3: 完全滤波后')
        plt.xlabel('时间 (秒)')
        plt.ylabel('磁场强度 (pT)')
        plt.legend()
        plt.grid(True)
    else:
        print(f"警告 ({channel_name}): 完全滤波数据为空。")

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 为 suptitle 留出空间
    
    if output_dir and base_filename:
        try:
            filename = f"{base_filename}_{channel_name.replace(' ', '_')}_signal_comparison.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        except Exception as e:
            print(f"错误：无法保存信号对比图 {filename}。原因: {e}")
    
    plt.show()
    plt.close(fig) # 保存后关闭图形

def find_and_plot_r_peaks(data, time_axis, channel_name, fs, min_height_factor, min_distance_ms,
                          output_dir=None, base_filename=""):
    """
    在信号中查找R峰并绘制标记，并保存。
    """
    if data is None or len(data) == 0:
        print(f"警告 ({channel_name}): R峰检测的输入数据为空。")
        return np.array([])

    min_h = np.max(data) * min_height_factor
    if min_h <= 0:
        data_std = np.std(data)
        min_h = data_std if data_std > 1e-9 else 1e-3
    
    min_dist_samples = int((min_distance_ms / 1000.0) * fs)

    print(f"\nR峰检测参数 ({channel_name}):")
    print(f"  最小高度因子: {min_height_factor*100:.1f}% -> 计算得到的最小高度阈值: {min_h:.3f} pT")
    print(f"  最小峰间距: {min_distance_ms} ms -> {min_dist_samples} 个采样点")

    height_param = min_h if min_h > 1e-9 else None

    try:
        peaks_indices, _ = find_peaks(data, height=height_param, distance=min_dist_samples)
    except Exception as e:
        print(f"错误 ({channel_name}): 调用 find_peaks 时出错: {e}")
        print(f"  使用的参数: height={height_param}, distance={min_dist_samples}")
        return np.array([])

    if len(peaks_indices) == 0:
        print(f"警告: 在 {channel_name} 未检测到R峰。请检查信号形态或调整该通道的R峰检测参数 (高度因子/距离)。")
    else:
        print(f"在 {channel_name} 检测到 {len(peaks_indices)} 个R峰。")

    fig_r_peaks = plt.figure(figsize=(15, 6)) # 获取figure对象
    plt.plot(time_axis, data, label='滤波后信号', linewidth=0.8)
    plt.plot(time_axis[peaks_indices], data[peaks_indices], "x", color='red', markersize=8, label=f'检测到的R峰 ({len(peaks_indices)}个)')
    plt.title(f'{channel_name} - R峰检测结果 (Height Factor: {min_height_factor:.2f}, Min Dist: {min_distance_ms}ms)')
    plt.xlabel('时间 (秒)')
    plt.ylabel('磁场强度 (pT)')
    plt.legend()
    plt.grid(True)
    
    if output_dir and base_filename:
        try:
            filename = f"{base_filename}_{channel_name.replace(' ', '_')}_R_peaks.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        except Exception as e:
            print(f"错误：无法保存R峰检测图 {filename}。原因: {e}")

    plt.show()
    plt.close(fig_r_peaks) # 保存后关闭图形
    
    return peaks_indices

def extract_and_plot_cycles_from_r_peaks(data, r_peaks_indices, time_axis_full_signal, channel_name, fs, 
                                         pre_r_ms, post_r_ms, 
                                         num_individual_cycles_to_plot,
                                         output_dir=None, base_filename=""):
    """
    根据R峰位置提取、绘制单个心跳周期，并计算和绘制平均心跳周期，并保存。
    """
    if data is None or r_peaks_indices is None or len(r_peaks_indices) == 0:
        print(f"警告 ({channel_name}): 没有有效的R峰或数据来提取周期。")
        return None

    pre_r_samples = int((pre_r_ms / 1000.0) * fs)
    post_r_samples = int((post_r_ms / 1000.0) * fs)
    total_cycle_samples = pre_r_samples + post_r_samples
    
    if total_cycle_samples <= 0:
        print(f"错误 ({channel_name}): 提取的周期长度 ({total_cycle_samples} 点) 无效。请检查 pre/post_r_ms 设置。")
        return None

    print(f"\n提取与平均心跳周期 ({channel_name}):")
    print(f"  R峰前提取: {pre_r_samples} 个点 ({pre_r_ms} ms)")
    print(f"  R峰后提取: {post_r_samples} 个点 ({post_r_ms} ms)")
    print(f"  总周期长度: {total_cycle_samples} 个点 ({pre_r_ms + post_r_ms} ms)")

    all_extracted_cycles = []
    cycle_time_axis_centered_at_r = np.linspace(-pre_r_ms / 1000.0, (post_r_samples -1) / 1000.0, total_cycle_samples)

    plot_individual = num_individual_cycles_to_plot > 0
    fig_individual = None # 初始化fig_individual
    num_plotted_individual = 0

    if plot_individual:
        fig_individual = plt.figure(figsize=(12, 7))
        plt.title(f'{channel_name} - 提取的单个心跳周期 (前 {num_individual_cycles_to_plot} 个, R峰在0s)')
        plt.xlabel('相对于R峰的时间 (秒)')
        plt.ylabel('磁场强度 (pT)')
        plt.grid(True)

    for i, r_peak_idx in enumerate(r_peaks_indices):
        start_idx = r_peak_idx - pre_r_samples
        end_idx = r_peak_idx + post_r_samples

        if start_idx >= 0 and end_idx <= len(data):
            cycle_data = data[start_idx:end_idx]
            if len(cycle_data) == total_cycle_samples:
                all_extracted_cycles.append(cycle_data)
                if plot_individual and num_plotted_individual < num_individual_cycles_to_plot:
                    r_peak_original_time = time_axis_full_signal[r_peak_idx] if r_peak_idx < len(time_axis_full_signal) else "未知时间"
                    label_text = f'周期 {i+1} (原R峰 @ {r_peak_original_time:.2f}s)' if isinstance(r_peak_original_time, float) else f'周期 {i+1}'
                    plt.plot(cycle_time_axis_centered_at_r, cycle_data, 
                             label=label_text if num_plotted_individual < 5 else None) # 最多显示5个图例
                    num_plotted_individual += 1
            else:
                print(f"警告 ({channel_name}): 周期 {i+1} @ R峰索引 {r_peak_idx} 长度不匹配 ({len(cycle_data)} vs {total_cycle_samples}), 已跳过。")
    
    if plot_individual and fig_individual is not None: # 确保fig_individual已创建
        if num_plotted_individual == 0 and len(all_extracted_cycles) == 0 :
            print(f"警告 ({channel_name}): 未能提取任何完整周期用于单独绘制。")
            plt.close(fig_individual) 
        elif num_plotted_individual > 0:
            plt.legend(loc='best')
            if output_dir and base_filename:
                try:
                    filename = f"{base_filename}_{channel_name.replace(' ', '_')}_individual_cycles.png"
                    save_path = os.path.join(output_dir, filename)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"图片已保存到: {save_path}")
                except Exception as e:
                    print(f"错误：无法保存单个周期图 {filename}。原因: {e}")
            plt.show()
            plt.close(fig_individual)
        else: 
            plt.close(fig_individual)

    if not all_extracted_cycles:
        print(f"信息 ({channel_name}): 没有足够的周期数据进行平均。")
        return None

    cycles_array = np.array(all_extracted_cycles)
    averaged_cycle = np.mean(cycles_array, axis=0)
    std_dev_cycle = np.std(cycles_array, axis=0)

    print(f"信息 ({channel_name}): 共使用了 {len(all_extracted_cycles)} 个周期进行平均。")

    fig_avg = plt.figure(figsize=(12, 7)) # 获取figure对象
    plt.title(f'{channel_name} - 平均心跳周期 (基于 {len(all_extracted_cycles)} 个周期, R峰在0s)')
    plt.xlabel('相对于R峰的时间 (秒)')
    plt.ylabel('磁场强度 (pT)')
    plt.grid(True)

    num_bg_cycles_to_plot = min(len(cycles_array), 50)
    indices_to_plot = np.random.choice(len(cycles_array), num_bg_cycles_to_plot, replace=False) if len(cycles_array) > num_bg_cycles_to_plot else np.arange(len(cycles_array))
    for idx in indices_to_plot:
        plt.plot(cycle_time_axis_centered_at_r, cycles_array[idx, :], color='gray', alpha=0.2, linewidth=0.8)

    plt.plot(cycle_time_axis_centered_at_r, averaged_cycle, color='red', linewidth=2, label='平均周期')
    plt.fill_between(cycle_time_axis_centered_at_r, 
                       averaged_cycle - std_dev_cycle, 
                       averaged_cycle + std_dev_cycle, 
                       color='red', alpha=0.15, label='平均值 ± 1 标准差')

    plt.legend(loc='best')
    
    if output_dir and base_filename:
        try:
            filename = f"{base_filename}_{channel_name.replace(' ', '_')}_averaged_cycle.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        except Exception as e:
            print(f"错误：无法保存平均周期图 {filename}。原因: {e}")
            
    plt.show()
    plt.close(fig_avg) # 保存后关闭图形
    
    return averaged_cycle

# --- 主程序 ---
if __name__ == "__main__":
    print(f"开始处理文件: {FILE_PATH}")
    # 确保输出目录存在
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)
    # 从文件路径中提取基本文件名，用于保存图片
    file_base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
    
    ch1_data_raw, ch2_data_raw = load_cardiac_data(FILE_PATH, SKIP_HEADER_LINES, FILE_ENCODING)

    INVERT_CHANNEL_1 = False 
    INVERT_CHANNEL_2 = False 

    active_channels_data = []
    channel_names_map = [] # 使用列表存储通道名，以保持顺序
    r_peak_params = {} 

    if ch1_data_raw is not None:
        if INVERT_CHANNEL_1:
            ch1_data_raw = -ch1_data_raw
            print("信息: Bx的信号已反转。")
        active_channels_data.append(ch1_data_raw)
        channel_names_map.append("Bx")
        r_peak_params["Bx"] = {
            'height_factor': R_PEAK_MIN_HEIGHT_FACTOR_CH1,
            'distance_ms': R_PEAK_MIN_DISTANCE_MS_CH1
        }
        
    if ch2_data_raw is not None:
        if INVERT_CHANNEL_2:
            ch2_data_raw = -ch2_data_raw
            print("信息: By的信号已反转。")
        active_channels_data.append(ch2_data_raw)
        channel_names_map.append("By")
        r_peak_params["By"] = {
            'height_factor': R_PEAK_MIN_HEIGHT_FACTOR_CH2,
            'distance_ms': R_PEAK_MIN_DISTANCE_MS_CH2
        }

    if not active_channels_data:
        print("错误: 没有成功加载任何通道的数据。程序终止。")
    else:
        all_averaged_cycles = {} 

        for i, raw_data in enumerate(active_channels_data):
            channel_name = channel_names_map[i] # 从列表中获取通道名
            print(f"\n--- 开始处理 {channel_name} ---")

            if len(raw_data) == 0:
                print(f"错误 ({channel_name}): 原始数据为空。")
                continue

            time_vector = np.arange(len(raw_data)) / SAMPLING_RATE

            data_bandpassed = apply_bandpass_filter(raw_data, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
            data_filtered = apply_notch_filter(data_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
            
            if data_filtered is not None and len(data_filtered) > 0:
                plot_signals_comparison(time_vector, raw_data, data_bandpassed, data_filtered, channel_name,
                                        output_dir=OUTPUT_FIGURE_DIR, base_filename=file_base_name)
            else:
                print(f"警告 ({channel_name}): 滤波后数据为空或无效，跳过信号对比图和后续步骤。")
                continue
            
            params = r_peak_params.get(channel_name)
            if params:
                r_peaks_indices = find_and_plot_r_peaks(
                    data_filtered, 
                    time_vector, 
                    channel_name, 
                    SAMPLING_RATE,
                    min_height_factor=params['height_factor'], 
                    min_distance_ms=params['distance_ms'],
                    output_dir=OUTPUT_FIGURE_DIR, base_filename=file_base_name
                )
            else:
                print(f"警告: 未找到通道 '{channel_name}' 的R峰参数，跳过R峰检测。")
                r_peaks_indices = np.array([])

            if r_peaks_indices is not None and len(r_peaks_indices) > 0:
                averaged_cycle = extract_and_plot_cycles_from_r_peaks(
                    data_filtered, r_peaks_indices, time_vector, channel_name, SAMPLING_RATE,
                    pre_r_ms=PRE_R_PEAK_MS, post_r_ms=POST_R_PEAK_MS, 
                    num_individual_cycles_to_plot=NUM_INDIVIDUAL_CYCLES_TO_PLOT,
                    output_dir=OUTPUT_FIGURE_DIR, base_filename=file_base_name
                )
                if averaged_cycle is not None:
                    all_averaged_cycles[channel_name] = averaged_cycle
                    print(f"信息 ({channel_name}): 已成功生成平均心跳周期。")
                else:
                    print(f"信息 ({channel_name}): 未能生成平均心跳周期。")
            else:
                print(f"信息 ({channel_name}): 未检测到R峰或R峰数量不足，无法提取或平均心跳周期。")
        
        print("\n--- 所有通道处理完毕 ---")
