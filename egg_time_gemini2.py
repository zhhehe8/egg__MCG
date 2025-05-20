import numpy as np
import zhplot
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import matplotlib
import os # Import os module for path operations

# 尝试设置后端，如果一个不行可以尝试其他，或者注释掉让matplotlib自动选择
# matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')

# 如果需要中文显示，取消注释下面两行
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# --- 配置参数 (用户可根据需要调整) ---
SAMPLING_RATE = 1000  # Hz, 从数据文件得知是1KC:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20
# FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d2\egg_d2_B3_t2.txt' # 你的数据文件路径
# FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d14\egg_d14_B28_t3.txt'
# FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d15\egg_d15_B29_t2.txt'
# FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d16\egg_d16_B27_t3.txt'
# FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d17\egg_d17_B29_t2.txt'
# FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d18\egg_d18_B33_t2.txt'
# FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t3.txt'
FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\朱鹮_250426\朱鹮day25_t3.txt'



SKIP_HEADER_LINES = 3 # 数据文件头部需要跳过的行数
FILE_ENCODING = 'utf-8' # 数据文件编码，通常是 'utf-8' 或 'gbk'
OUTPUT_FIGURE_DIR = r'C:\Users\Xiaoning Tan\Desktop\egg_figure' # *** 图片保存目录 ***

# 滤波参数 (通用)
FILTER_ORDER_BANDPASS = 3 # 带通滤波器的阶数
LOWCUT_FREQ = 0.5         # Hz, 低截止频率
HIGHCUT_FREQ = 45.0       # Hz, 高截止频率
NOTCH_FREQ_MAINS = 50.0   # Hz, 工频干扰频率
Q_FACTOR_NOTCH = 30.0     # 陷波滤波器的品质因数

# ----------------------------------------------------------
# R峰检测参数 - 针对融合后的信号 <- *** 需要重点调整 ***
# ----------------------------------------------------------
R_PEAK_MIN_HEIGHT_FACTOR_COMBINED = 0.3 # 范围建议 0.2 - 0.6，需要根据Signal调整
R_PEAK_MIN_DISTANCE_MS_COMBINED = 150
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
                                     output_dir=None, # 新增：输出目录参数
                                     base_filename=""): # 新增：用于保存的文件名前缀
    """
    根据R峰位置提取周期，计算、绘制并保存平均心跳周期图。
    """
    if data is None or r_peaks_indices is None or len(r_peaks_indices) < 2:
        print(f"警告 ({channel_name}): 没有足够的有效R峰 ({len(r_peaks_indices) if r_peaks_indices is not None else 0}个) 或数据来提取和平均周期。")
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
        print(f"信息 ({channel_name}): 没有足够的有效周期数据 ({valid_cycles_count}个) 进行平均。")
        return None

    cycles_array = np.array(all_extracted_cycles)
    averaged_cycle = np.mean(cycles_array, axis=0)
    std_dev_cycle = np.std(cycles_array, axis=0)

    print(f"信息 ({channel_name}): 共使用了 {valid_cycles_count} 个有效周期进行平均。")

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(12, 7)) # 调整图像大小，获取ax对象
    ax.set_title(f'{channel_name} - Averaged Waveforms (Based on {valid_cycles_count} cycles)', fontsize=16) # 英文标题
    ax.set_xlabel('Time (s)', fontsize=12) # 更新X轴标签
    ax.set_ylabel('Signal Magnitude (pT)', fontsize=12) # 更新Y轴标签
    ax.grid(True, which='major', linestyle='-', linewidth='0.8', color='grey') # 主要网格
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgrey') # 次要网格
    ax.minorticks_on() # 开启次要刻度

    # 绘制背景中的单个周期 (灰色)
    num_bg_cycles_to_plot = min(len(cycles_array), 30)
    indices_to_plot = np.random.choice(len(cycles_array), num_bg_cycles_to_plot, replace=False) if len(cycles_array) > num_bg_cycles_to_plot else np.arange(len(cycles_array))
    for idx in indices_to_plot:
        ax.plot(cycle_time_axis_centered_at_r, cycles_array[idx, :], color='lightgray', alpha=0.3, linewidth=0.6) # 调整透明度和线宽

    # 绘制平均周期和标准差 (使用新颜色)，英文标题
    ax.plot(cycle_time_axis_centered_at_r, averaged_cycle, color='orangered', linewidth=2.5, label='Averaged Cycle') # 橙红色
    ax.fill_between(cycle_time_axis_centered_at_r,
                       averaged_cycle - std_dev_cycle,
                       averaged_cycle + std_dev_cycle,
                       color='lightcoral', alpha=0.3, label='Averaged ± 1 Standard Deviation') # 浅珊瑚色填充

    ax.legend(loc='best')
    plt.tight_layout()

    # --- 保存图片 ---
    if output_dir:
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            # 构建保存路径
            save_filename = f"{base_filename}_avg_cycle.png"
            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        except Exception as e:
            print(f"错误：无法保存图片到 {output_dir}。原因: {e}")
    else:
        print("警告：未指定输出目录，图片将不会被保存。")

    plt.show() # 显示图片

    return averaged_cycle

# --- 主程序 ---
if __name__ == "__main__":
    print(f"开始处理文件: {FILE_PATH}")
    # 从文件路径中提取基本文件名，用于保存图片
    file_base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]

    # 1. 加载数据
    ch1_data_raw, ch2_data_raw = load_cardiac_data(FILE_PATH, SKIP_HEADER_LINES, FILE_ENCODING)

    if ch1_data_raw is None or ch2_data_raw is None:
        print("错误: 未能成功加载两个通道的数据。程序终止。")
    else:
        print("\n--- 开始滤波 ---")
        # 2. 独立滤波
        ch1_bandpassed = apply_bandpass_filter(ch1_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
        ch1_filtered = apply_notch_filter(ch1_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
        print("通道 1 (Bx) 滤波完成。")

        ch2_bandpassed = apply_bandpass_filter(ch2_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
        ch2_filtered = apply_notch_filter(ch2_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
        print("通道 2 (By) 滤波完成。")

        if ch1_filtered is None or ch2_filtered is None:
             print("错误：一个或两个通道滤波失败。无法继续。")
        else:
            # 3. 数据融合 (计算幅度)
            print("\n--- 开始Signal ---")
            min_len = min(len(ch1_filtered), len(ch2_filtered))
            combined_data_filtered = np.sqrt(ch1_filtered[:min_len]**2 + ch2_filtered[:min_len]**2)
            print(f"信号融合完成，生成Signal长度: {len(combined_data_filtered)}")

            time_vector = np.arange(len(combined_data_filtered)) / SAMPLING_RATE

            # 4. 在Signal上查找R峰
            r_peaks_indices = find_r_peaks(
                combined_data_filtered,
                SAMPLING_RATE,
                min_height_factor=R_PEAK_MIN_HEIGHT_FACTOR_COMBINED,
                min_distance_ms=R_PEAK_MIN_DISTANCE_MS_COMBINED,
                channel_name="Signal"
            )

            # 5. 提取、绘制并保存平均心跳周期
            if r_peaks_indices is not None and len(r_peaks_indices) > 0:
                averaged_cycle = extract_and_plot_average_cycle(
                    combined_data_filtered, r_peaks_indices, SAMPLING_RATE,
                    pre_r_ms=PRE_R_PEAK_MS, post_r_ms=POST_R_PEAK_MS,
                    channel_name="Signal",
                    output_dir=OUTPUT_FIGURE_DIR, # 传递输出目录
                    base_filename=file_base_name # 传递基本文件名
                )
                if averaged_cycle is not None:
                    print(f"\n信息 (Signal): 已成功生成、绘制并尝试保存平均心跳周期。")
                    # np.savetxt(os.path.join(OUTPUT_FIGURE_DIR, f'{file_base_name}_avg_cycle_data.txt'), averaged_cycle) # 可选：保存平均周期数据
                else:
                    print(f"\n信息 (Signal): 未能生成平均心跳周期。")
            else:
                print(f"\n信息 (Signal): 未检测到R峰或R峰数量不足，无法提取或平均心跳周期。")

    print("\n--- 处理结束 ---")
