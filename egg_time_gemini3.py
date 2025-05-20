"""   将多个数据的平均R峰绘制到同一张图上  """

import zhplot 
# 尝试使用Qt5Agg后端以获得更好的交互性，如果不行可尝试 'TkAgg'
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import os # 导入os模块用于路径操作

# --- 配置参数 (用户可根据需要调整) ---
SAMPLING_RATE = 1000  # Hz, 从数据文件得知是1K

# *** 修改这里：包含所有需要处理的文件路径列表 ***
FILE_PATHS = [
    # r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d12\egg_d12_B20_t2.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d13\egg_d13_B20_t2.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d14\egg_d14_B27_t1.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d15\egg_d15_B27_t2.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d16\egg_d16_B27_t1.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d17\egg_d17_B29_t3.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d18\egg_d18_B33_t2.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d19\egg_d19_B30_t1.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t3.txt',
    
    # 添加更多文件路径，直到总共9个或您需要的数量
    # r'C:\path\to\your\file8.txt',
    # r'C:\path\to\your\file9.txt',
]

SKIP_HEADER_LINES = 2 # 数据文件头部需要跳过的行数
FILE_ENCODING = 'utf-8' # 数据文件编码，通常是 'utf-8' 或 'gbk'
OUTPUT_FIGURE_DIR = r'C:\Users\Xiaoning Tan\Desktop\egg_figure' # *** 图片保存目录 ***

# 滤波参数 (通用)
FILTER_ORDER_BANDPASS = 4 # 带通滤波器的阶数
LOWCUT_FREQ = 0.5         # Hz, 低截止频率
HIGHCUT_FREQ = 45.0       # Hz, 高截止频率
NOTCH_FREQ_MAINS = 50.0   # Hz, 工频干扰频率
Q_FACTOR_NOTCH = 30.0     # 陷波滤波器的品质因数

# ----------------------------------------------------------
# R峰检测参数 - 针对融合后的信号 <- *** 需要重点调整 ***
# ----------------------------------------------------------
R_PEAK_MIN_HEIGHT_FACTOR_COMBINED = 0.4 # 范围建议 0.2 - 0.6，需要根据Signal调整
R_PEAK_MIN_DISTANCE_MS_COMBINED = 200
# ----------------------------------------------------------

# 周期提取和平均参数 (通用)
PRE_R_PEAK_MS = 100       # ms, 提取周期时R峰前的时间
POST_R_PEAK_MS = 150      # ms, 提取周期时R峰后的时间

# --- 函数定义 ---

def load_cardiac_data(filepath, skip_header, file_encoding='utf-8'):
    """
    从文本文件加载胚鸡心磁数据，预期至少有两列。
    """
    try:
        data = np.loadtxt(filepath, skiprows=skip_header, dtype=float, encoding=file_encoding)
        if data.ndim == 1 or data.shape[1] < 2:
            print(f"错误：数据文件 {filepath} 需要至少两列 (Bx, By)。当前列数: {data.shape[1] if data.ndim > 1 else 1}")
            return None, None
        channel1 = data[:, 0]
        channel2 = data[:, 1]
        print(f"成功加载文件: {filepath}, 数据点数: {len(channel1)}")
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
    if high >= 1.0:
        high = 0.99
        print(f"警告: highcut频率({highcut}Hz)过高,已调整为奈奎斯特频率的99%。")
    if low <= 0:
        low = 0.001
        print(f"警告: lowcut频率({lowcut}Hz)过低,已调整为0.001*奈奎斯特频率。")
    if low >= high:
        print(f"错误: 带通滤波器的低截止({lowcut}Hz)必须小于高截止({highcut}Hz)。跳过带通滤波。")
        return data

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

def find_r_peaks(data, fs, min_height_factor, min_distance_ms, channel_name="信号"):
    """
    在信号中查找R峰。返回峰值索引。
    """
    if data is None or len(data) == 0:
        print(f"警告 ({channel_name}): R峰检测的输入数据为空。")
        return np.array([])

    data_max = np.max(data)
    if data_max <= 0:
        data_max = np.percentile(data, 99)
        if data_max <= 0:
            print(f"警告 ({channel_name}): 信号最大值为非正数，无法计算高度阈值。")
            return np.array([])

    min_h = data_max * min_height_factor

    if min_h <= 1e-9:
        data_std = np.std(data)
        min_h = data_std * 0.5 if data_std > 1e-9 else 1e-3
        print(f"警告 ({channel_name}): 计算得到的最小高度阈值过低或无效，已调整为: {min_h:.3f}")

    min_dist_samples = int((min_distance_ms / 1000.0) * fs)

    print(f"\nR峰检测参数 ({channel_name}):")
    print(f"  最小高度因子: {min_height_factor*100:.1f}% (基于信号最大值 {data_max:.3f}) -> 计算得到的最小高度阈值: {min_h:.3f}")
    print(f"  最小峰间距: {min_distance_ms} ms -> {min_dist_samples} 个采样点")

    height_param = min_h if min_h > 1e-9 else None

    try:
        peaks_indices, properties = find_peaks(data, height=height_param, distance=min_dist_samples)
    except Exception as e:
        print(f"错误 ({channel_name}): 调用 find_peaks 时出错: {e}")
        print(f"  使用的参数: height={height_param}, distance={min_dist_samples}")
        return np.array([])

    if len(peaks_indices) == 0:
        print(f"警告: 在 {channel_name} 未检测到R峰。请检查信号形态或调整该信号的R峰检测参数 (高度因子/距离)。")
    else:
        print(f"在 {channel_name} 检测到 {len(peaks_indices)} 个R峰。")

    return peaks_indices

def extract_and_plot_average_cycle(data, r_peaks_indices, fs,
                                     pre_r_ms, post_r_ms,
                                     channel_name="信号",
                                     output_dir=None, 
                                     base_filename=""):
    """
    根据R峰位置提取周期，计算、绘制并保存平均心跳周期图。
    返回平均周期数据和对应的时间轴。
    """
    if data is None or r_peaks_indices is None or len(r_peaks_indices) < 2:
        print(f"警告 ({channel_name} for {base_filename}): 没有足够的有效R峰 ({len(r_peaks_indices) if r_peaks_indices is not None else 0}个) 或数据来提取和平均周期。")
        return None, None

    pre_r_samples = int((pre_r_ms / 1000.0) * fs)
    post_r_samples = int((post_r_ms / 1000.0) * fs)
    total_cycle_samples = pre_r_samples + post_r_samples

    if total_cycle_samples <= 0:
        print(f"错误 ({channel_name} for {base_filename}): 提取的周期长度 ({total_cycle_samples} 点) 无效。请检查 pre/post_r_ms 设置。")
        return None, None

    print(f"\n提取与平均心跳周期 ({channel_name} for {base_filename}):")
    print(f"  R峰前提取: {pre_r_samples} 个点 ({pre_r_ms} ms)")
    print(f"  R峰后提取: {post_r_samples} 个点 ({post_r_ms} ms)")
    print(f"  总周期长度: {total_cycle_samples} 个点 ({pre_r_ms + post_r_ms} ms)")

    all_extracted_cycles = []
    cycle_time_axis_centered_at_r = np.linspace(-pre_r_ms / 1000.0, (post_r_samples -1) / 1000.0, total_cycle_samples)

    valid_cycles_count = 0
    for i, r_peak_idx in enumerate(r_peaks_indices):
        start_idx = r_peak_idx - pre_r_samples
        end_idx = r_peak_idx + post_r_samples
        if start_idx >= 0 and end_idx <= len(data):
            cycle_data = data[start_idx:end_idx]
            if len(cycle_data) == total_cycle_samples:
                all_extracted_cycles.append(cycle_data)
                valid_cycles_count += 1

    if valid_cycles_count < 2:
        print(f"信息 ({channel_name} for {base_filename}): 没有足够的有效周期数据 ({valid_cycles_count}个) 进行平均。")
        return None, None

    cycles_array = np.array(all_extracted_cycles)
    averaged_cycle = np.mean(cycles_array, axis=0)
    std_dev_cycle = np.std(cycles_array, axis=0)

    print(f"信息 ({channel_name} for {base_filename}): 共使用了 {valid_cycles_count} 个有效周期进行平均。")

    # --- 绘图 (单个文件的平均周期) ---
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.set_title(f'Averaged Waveform: {base_filename} (Combined Signal)\n(Based on {valid_cycles_count} cycles)', fontsize=14)
    ax.set_xlabel('Time relative to R-peak (s)', fontsize=12) 
    ax.set_ylabel('Signal Magnitude (pT)', fontsize=12) 
    ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='grey') 
    ax.grid(True, which='minor', linestyle=':', linewidth='0.4', color='lightgrey') 
    ax.minorticks_on() 

    num_bg_cycles_to_plot = min(len(cycles_array), 20) # 减少背景数量
    indices_to_plot = np.random.choice(len(cycles_array), num_bg_cycles_to_plot, replace=False) if len(cycles_array) > num_bg_cycles_to_plot else np.arange(len(cycles_array))
    for idx in indices_to_plot:
        ax.plot(cycle_time_axis_centered_at_r, cycles_array[idx, :], color='lightgray', alpha=0.35, linewidth=0.7)

    ax.plot(cycle_time_axis_centered_at_r, averaged_cycle, color='orangered', linewidth=2, label='Averaged Cycle') 
    ax.fill_between(cycle_time_axis_centered_at_r,
                       averaged_cycle - std_dev_cycle,
                       averaged_cycle + std_dev_cycle,
                       color='lightcoral', alpha=0.35, label='Averaged ± 1 Std Dev') 

    ax.legend(loc='best')
    plt.tight_layout()

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            save_filename = f"{base_filename}_combined_avg_cycle.png" # 明确是combined signal的
            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        except Exception as e:
            print(f"错误：无法保存图片到 {output_dir}。原因: {e}")
    
    plt.show() 
    plt.close(fig) # 关闭单个平均图

    return averaged_cycle, cycle_time_axis_centered_at_r # 返回时间和数据

def plot_all_average_cycles_on_one_graph(all_cycles_data, output_dir=None):
    """
    将所有文件处理得到的平均心跳周期绘制在一张图上。
    all_cycles_data: 一个列表，每个元素是 (label, time_axis, averaged_cycle_data)
    """
    if not all_cycles_data:
        print("没有可供比较的平均周期数据。")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_title('胚鸡平均心跳周期（13-20日龄）', fontsize=16)
    ax.set_xlabel('Time relative to R-peak (s)', fontsize=12)
    ax.set_ylabel('Signal Magnitude (pT)', fontsize=12)
    ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='grey')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.4', color='lightgrey')
    ax.minorticks_on()

    # 使用matplotlib的颜色循环
    colors = plt.cm.get_cmap('tab10', len(all_cycles_data)) # 'tab10' 'viridis' 'plasma'

    for i, (label, time_axis, avg_cycle) in enumerate(all_cycles_data):
        if time_axis is not None and avg_cycle is not None:
            ax.plot(time_axis, avg_cycle, linewidth=1.5, label=label, color=colors(i))
        else:
            print(f"跳过绘制: {label} 因为数据不完整。")

    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            comparison_save_path = os.path.join(output_dir, "all_files_compared_average_cycles.png")
            plt.savefig(comparison_save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存到: {comparison_save_path}")
        except Exception as e:
            print(f"错误：无法保存对比图。原因: {e}")
            
    plt.show()
    plt.close(fig)


# --- 主程序 ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True) # 确保输出目录存在
    
    all_averaged_cycles_for_comparison = [] # 用于存储所有文件的平均周期数据和标签

    for current_file_path in FILE_PATHS:
        print(f"\n--- 开始处理文件: {current_file_path} ---")
        file_base_name = os.path.splitext(os.path.basename(current_file_path))[0]

        ch1_data_raw, ch2_data_raw = load_cardiac_data(current_file_path, SKIP_HEADER_LINES, FILE_ENCODING)

        if ch1_data_raw is None or ch2_data_raw is None:
            print(f"错误: 未能成功加载文件 {current_file_path} 的两个通道数据。跳过此文件。")
            continue
        
        print("\n  --- 开始滤波 ---")
        ch1_bandpassed = apply_bandpass_filter(ch1_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
        ch1_filtered = apply_notch_filter(ch1_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
        print(f"  通道 1 (Bx) from {file_base_name} 滤波完成。")

        ch2_bandpassed = apply_bandpass_filter(ch2_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
        ch2_filtered = apply_notch_filter(ch2_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
        print(f"  通道 2 (By) from {file_base_name} 滤波完成。")

        if ch1_filtered is None or ch2_filtered is None:
            print(f"错误：文件 {file_base_name} 的一个或两个通道滤波失败。无法继续处理此文件。")
            continue
        
        print("\n  --- 开始信号融合 ---")
        min_len = min(len(ch1_filtered), len(ch2_filtered))
        combined_data_filtered = np.sqrt(ch1_filtered[:min_len]**2 + ch2_filtered[:min_len]**2)
        print(f"  信号融合完成 ({file_base_name})，生成融合信号长度: {len(combined_data_filtered)}")

        # time_vector = np.arange(len(combined_data_filtered)) / SAMPLING_RATE # 这个时间轴是针对整个信号的

        r_peaks_indices = find_r_peaks(
            combined_data_filtered,
            SAMPLING_RATE,
            min_height_factor=R_PEAK_MIN_HEIGHT_FACTOR_COMBINED,
            min_distance_ms=R_PEAK_MIN_DISTANCE_MS_COMBINED,
            channel_name=f"融合信号 ({file_base_name})"
        )

        if r_peaks_indices is not None and len(r_peaks_indices) > 0:
            # 注意：extract_and_plot_average_cycle 现在返回 (averaged_cycle, cycle_time_axis_centered_at_r)
            averaged_cycle_data, cycle_time_axis = extract_and_plot_average_cycle(
                combined_data_filtered, r_peaks_indices, SAMPLING_RATE,
                pre_r_ms=PRE_R_PEAK_MS, post_r_ms=POST_R_PEAK_MS,
                channel_name=f"融合信号 ({file_base_name})",
                output_dir=OUTPUT_FIGURE_DIR, 
                base_filename=file_base_name 
            )
            if averaged_cycle_data is not None and cycle_time_axis is not None:
                # 使用文件名作为标签，去除扩展名
                label_for_plot = os.path.splitext(os.path.basename(current_file_path))[0]
                all_averaged_cycles_for_comparison.append((label_for_plot, cycle_time_axis, averaged_cycle_data))
                print(f"  信息 ({file_base_name}): 已成功生成平均心跳周期并添加到对比列表。")
            else:
                print(f"  信息 ({file_base_name}): 未能生成平均心跳周期。")
        else:
            print(f"  信息 ({file_base_name}): 未检测到R峰或R峰数量不足，无法提取或平均心跳周期。")

    # 所有文件处理完毕后，绘制对比图
    if all_averaged_cycles_for_comparison:
        print("\n--- 开始绘制所有文件的平均周期对比图 ---")
        plot_all_average_cycles_on_one_graph(all_averaged_cycles_for_comparison, output_dir=OUTPUT_FIGURE_DIR)
    else:
        print("\n没有成功处理任何文件以生成平均周期对比图。")

    print("\n--- 批量处理结束 ---")
