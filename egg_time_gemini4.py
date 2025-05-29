"""
    批量处理信号数据，求出平均心跳周期
"""



import numpy as np
import zhplot
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import os
import re
import matplotlib.pyplot as plt
import matplotlib # 确保matplotlib被导入

# --- 配置参数 (用户可根据需要调整) ---
SAMPLING_RATE = 1000  # Hz

# *** 指向包含所有每日数据文件夹的根目录 ***
ROOT_DATA_DIR = r'C:\Users\Xiaoning Tan\Desktop\16+17day'
OUTPUT_FIGURE_DIR = r'C:\Users\Xiaoning Tan\Desktop\output_time' # *** 平均周期图片保存目录 ***

SKIP_HEADER_LINES = 2 # 数据文件头部需要跳过的行数 (根据用户最新提供)
FILE_ENCODING = 'utf-8' # 数据文件编码

# 滤波参数 (通用)
FILTER_ORDER_BANDPASS = 4 # 带通滤波器的阶数 (根据用户最新提供)
LOWCUT_FREQ = 0.5         # Hz, 低截止频率
HIGHCUT_FREQ = 45.0       # Hz, 高截止频率
NOTCH_FREQ_MAINS = 50.0   # Hz, 工频干扰频率
Q_FACTOR_NOTCH = 30.0     # 陷波滤波器的品质因数

# R峰检测参数 (通用 - 作为默认值)
DEFAULT_R_PEAK_MIN_HEIGHT_FACTOR = 0.4 # (根据用户最新提供)
DEFAULT_R_PEAK_MIN_DISTANCE_MS = 150   # (根据用户最新提供)

# 周期提取和平均参数 (通用)
PRE_R_PEAK_MS = 100       # ms, 提取周期时R峰前的时间
POST_R_PEAK_MS = 100      # ms, 提取周期时R峰后的时间 (根据用户最新提供)


# --- 日龄特定的R峰检测参数 ---
DAY_SPECIFIC_R_PEAK_PARAMS = {
    "Day 2":  {"height_factor": 0.35, "distance_ms": 400}, 
    "Day 3":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 4":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 5":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 6":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 7":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 8":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 9":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 10": {"height_factor": 0.35, "distance_ms": 200},
    "Day 11": {"height_factor": 0.35, "distance_ms": 300},
    "Day 12": {"height_factor": 0.35, "distance_ms": 300},
    "Day 13": {"height_factor": 0.40, "distance_ms": 300}, 
    "Day 14": {"height_factor": 0.40, "distance_ms": 250},
    "Day 15": {"height_factor": 0.40, "distance_ms": 250},
    "Day 16": {"height_factor": 0.40, "distance_ms": 250},
    "Day 17": {"height_factor": 0.40, "distance_ms": 250},
    "Day 18": {"height_factor": 0.40, "distance_ms": 200},
    "Day 19": {"height_factor": 0.40, "distance_ms": 200},
    "Day 20": {"height_factor": 0.45, "distance_ms": 200}, 
    "Day 21": {"height_factor": 0.45, "distance_ms": 200},
}


# --- 函数定义 ---

def extract_day_label_from_folder(folder_name):
    match_detailed = re.search(r'[Dd](\d+)', folder_name) 
    if match_detailed:
        return f"Day {match_detailed.group(1)}"
    match_simple_day = re.search(r'day(\d+)', folder_name, re.IGNORECASE)
    if match_simple_day:
        return f"Day {match_simple_day.group(1)}"
    print(f"警告：无法从文件夹 '{folder_name}' 中提取标准日龄标签。将使用文件夹名作为标签。")
    return folder_name


def load_cardiac_data(filepath, skip_header, file_encoding='utf-8'):
    try:
        data = np.loadtxt(filepath, skiprows=skip_header, dtype=float, encoding=file_encoding)
        if data.ndim == 1 or data.shape[1] < 2:
            print(f"  错误：数据文件 {os.path.basename(filepath)} 需要至少两列 (Bx, By)。")
            return None, None
        channel1 = data[:, 0]
        channel2 = data[:, 1]
        return channel1, channel2
    except Exception as e:
        print(f"  读取文件 '{os.path.basename(filepath)}' 或解析数据时出错: {e}")
        return None, None

def apply_bandpass_filter(data, lowcut, highcut, fs, order):
    if data is None: return None
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0: high = 0.99
    if low <= 0: low = 0.001 
    if low >= high: return data
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfiltfilt(sos, data)

def apply_notch_filter(data, notch_freq, quality_factor, fs):
    if data is None: return None
    nyquist = 0.5 * fs
    freq_normalized = notch_freq / nyquist
    if freq_normalized >= 1.0 or freq_normalized <= 0: return data
    b, a = iirnotch(freq_normalized, quality_factor)
    return filtfilt(b, a, data)

""" 后面需要修改 find_r_peaks_data"""
def find_r_peaks_data(data, fs, min_height_factor, min_distance_ms, identifier="信号",percentile=99):
    if data is None or len(data) == 0: return np.array([])
    data_max = np.max(data)
    if data_max <= 1e-9: 
        robust_max = np.percentile(data, 99)
        if robust_max <= 1e-9: return np.array([]) 
        min_h = robust_max * min_height_factor
    else:
        min_h = data_max * min_height_factor

    if min_h <= 1e-9: 
        data_std = np.std(data)
        min_h = data_std * 0.5 if data_std > 1e-9 else 1e-3
    
    min_dist_samples = int((min_distance_ms / 1000.0) * fs)
    height_param = min_h if min_h > 1e-9 else None
    
    try:
        peaks_indices, _ = find_peaks(data, height=height_param, distance=min_dist_samples)
    except Exception as e:
        print(f"  错误 ({identifier}): 调用 find_peaks 时出错: {e}")
        return np.array([])
    return peaks_indices

def generate_and_save_average_cycle_plot(data, r_peaks_indices, fs, 
                                         pre_r_ms, post_r_ms, 
                                         output_dir, base_filename, 
                                         identifier="信号"):
    if data is None or r_peaks_indices is None or len(r_peaks_indices) < 2:
        return False 

    pre_r_samples = int((pre_r_ms / 1000.0) * fs)
    post_r_samples = int((post_r_ms / 1000.0) * fs)
    total_cycle_samples = pre_r_samples + post_r_samples

    if total_cycle_samples <= 0:
        print(f"  错误 ({identifier}): 提取的周期长度无效 (pre: {pre_r_ms}ms, post: {post_r_ms}ms)。")
        return False

    all_extracted_cycles = []
    valid_cycles_count = 0
    for r_peak_idx in r_peaks_indices:
        start_idx = r_peak_idx - pre_r_samples
        end_idx = r_peak_idx + post_r_samples
        if start_idx >= 0 and end_idx <= len(data):
            cycle_data = data[start_idx:end_idx]
            if len(cycle_data) == total_cycle_samples:
                all_extracted_cycles.append(cycle_data)
                valid_cycles_count += 1

    if valid_cycles_count < 2:
        return False

    cycles_array = np.array(all_extracted_cycles)
    
    # --- 基线校正 ---
    # 1. 计算原始平均周期和标准差（用于确定标准差的幅度）
    original_averaged_cycle = np.mean(cycles_array, axis=0)
    std_dev_cycle = np.std(cycles_array, axis=0) # 标准差基于原始（未校正）的变异性

    # 2. 从原始平均周期中确定基线偏移量
    #    使用 PRE_R_PEAK_MS 的前30ms作为基线窗口，如果 PRE_R_PEAK_MS 不足30ms，则使用整个 PRE_R_PEAK_MS
    baseline_window_duration_ms = min(pre_r_ms, 30) 
    baseline_samples_count = int((baseline_window_duration_ms / 1000.0) * fs)
    if baseline_samples_count > 0 and baseline_samples_count <= pre_r_samples : # 确保基线窗口有效
        baseline_offset_for_avg = np.mean(original_averaged_cycle[:baseline_samples_count])
    else: # 如果窗口无效（例如pre_r_ms太短），则不进行偏移
        baseline_offset_for_avg = 0
        print(f"  警告 ({identifier}): 基线校正窗口无效 (pre_r_ms: {pre_r_ms}ms, calculated samples: {baseline_samples_count}). 平均周期未进行基线校正。")


    # 3. 校正平均周期
    averaged_cycle_corrected = original_averaged_cycle - baseline_offset_for_avg
    
    # 4. 校正背景中的单个周期（用于绘图）
    corrected_background_cycles = []
    for cycle in cycles_array:
        if baseline_samples_count > 0 and baseline_samples_count <= pre_r_samples:
            individual_baseline_offset = np.mean(cycle[:baseline_samples_count])
            corrected_background_cycles.append(cycle - individual_baseline_offset)
        else:
            corrected_background_cycles.append(cycle) # 如果窗口无效，不校正单个周期
    corrected_background_cycles = np.array(corrected_background_cycles)
    # --- 基线校正结束 ---


    cycle_time_axis_centered_at_r = np.linspace(-pre_r_ms / 1000.0, (post_r_samples -1) / 1000.0, total_cycle_samples)

    fig, ax = plt.subplots(figsize=(10, 6)) 
    plot_title = f'Averaged Cardiac Cycle: {base_filename}\n(Based on {valid_cycles_count} cycles, R-peak at 0s, Baseline Corrected)'
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Time relative to R-peak (s)', fontsize=12)
    ax.set_ylabel('Signal Magnitude (pT, Baseline Corrected)', fontsize=12)
    
    ax.set_ylim(0, 2) # 将Y轴范围固定为0到2

    ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='grey')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.4', color='lightgrey')
    ax.minorticks_on()

    num_bg_cycles_to_plot = min(len(corrected_background_cycles), 20) 
    indices_to_plot = np.random.choice(len(corrected_background_cycles), num_bg_cycles_to_plot, replace=False) if len(corrected_background_cycles) > num_bg_cycles_to_plot else np.arange(len(corrected_background_cycles))
    for idx in indices_to_plot:
        ax.plot(cycle_time_axis_centered_at_r, corrected_background_cycles[idx, :], color='lightgray', alpha=0.35, linewidth=0.7)

    # 绘制校正后的平均周期和原始标准差（围绕校正后的均值）
    ax.plot(cycle_time_axis_centered_at_r, averaged_cycle_corrected, color='orangered', linewidth=2, label='Averaged Cycle (Corrected)')
    ax.fill_between(cycle_time_axis_centered_at_r,
                       averaged_cycle_corrected - std_dev_cycle, # std_dev_cycle 未变，但中心线变了
                       averaged_cycle_corrected + std_dev_cycle,
                       color='lightcoral', alpha=0.35, label='Avg (Corrected) ± 1 Std Dev')
    ax.legend(loc='best')
    plt.tight_layout()

    try:
        output_image_filename = f"{base_filename}_avg_cycle_baseline_corrected.png" # 更新文件名
        output_image_path = os.path.join(output_dir, output_image_filename)
        
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"  成功：平均周期图片已保存到: {output_image_path}")
        plt.close(fig) 
        return True 
    except Exception as e:
        print(f"  错误：无法保存平均周期图片到 {output_image_path}。原因: {e}")
        plt.close(fig) 
        return False

# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.isdir(ROOT_DATA_DIR):
        print(f"错误：根数据目录 '{ROOT_DATA_DIR}' 不存在或不是一个目录。请检查路径。")
    else:
        print(f"开始批量处理根目录: {ROOT_DATA_DIR}")
        print(f"平均周期图片将保存到: {OUTPUT_FIGURE_DIR}")
        os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True) 
        
        processed_files_count = 0
        saved_avg_cycle_images_count = 0

        day_folders = sorted(
            [d for d in os.listdir(ROOT_DATA_DIR) if os.path.isdir(os.path.join(ROOT_DATA_DIR, d))],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf')
        )

        for day_folder_name in day_folders:
            day_folder_path = os.path.join(ROOT_DATA_DIR, day_folder_name)
            day_label = extract_day_label_from_folder(day_folder_name) 
            print(f"\n--- 开始处理文件夹: {day_folder_name} (日龄标签: {day_label}) ---")
            
            day_r_peak_config = DAY_SPECIFIC_R_PEAK_PARAMS.get(day_label, {})
            current_r_peak_height_factor = day_r_peak_config.get("height_factor", DEFAULT_R_PEAK_MIN_HEIGHT_FACTOR)
            current_r_peak_distance_ms = day_r_peak_config.get("distance_ms", DEFAULT_R_PEAK_MIN_DISTANCE_MS)
            
            print(f"  使用R峰检测参数: Height Factor = {current_r_peak_height_factor:.2f}, Distance MS = {current_r_peak_distance_ms}")

            current_output_subfolder = os.path.join(OUTPUT_FIGURE_DIR, day_folder_name)
            os.makedirs(current_output_subfolder, exist_ok=True)

            txt_files_in_day_folder = [f for f in os.listdir(day_folder_path) if f.lower().endswith('.txt')]
            if not txt_files_in_day_folder:
                print(f"  在文件夹 {day_folder_name} 中未找到 .txt 文件。")
                continue

            for txt_filename in txt_files_in_day_folder:
                current_file_path = os.path.join(day_folder_path, txt_filename)
                print(f"\n  处理文件: {txt_filename}")
                processed_files_count += 1
                
                ch1_data_raw, ch2_data_raw = load_cardiac_data(current_file_path, SKIP_HEADER_LINES, FILE_ENCODING)

                if ch1_data_raw is None or ch2_data_raw is None:
                    continue
                
                ch1_bandpassed = apply_bandpass_filter(ch1_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
                ch1_filtered = apply_notch_filter(ch1_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
                
                ch2_bandpassed = apply_bandpass_filter(ch2_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
                ch2_filtered = apply_notch_filter(ch2_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)

                if ch1_filtered is None or ch2_filtered is None:
                    print(f"  错误：文件 {txt_filename} 的一个或两个通道滤波失败。")
                    continue
                
                min_len = min(len(ch1_filtered), len(ch2_filtered))
                combined_data_filtered = np.sqrt(ch1_filtered[:min_len]**2 + ch2_filtered[:min_len]**2)
                
                r_peaks_indices = find_r_peaks_data(
                    combined_data_filtered, SAMPLING_RATE,
                    min_height_factor=current_r_peak_height_factor,
                    min_distance_ms=current_r_peak_distance_ms,
                    identifier=f"融合信号 ({txt_filename})"
                )

                if r_peaks_indices is not None and len(r_peaks_indices) > 1:
                    output_filename_base_for_plot = os.path.splitext(txt_filename)[0]
                    
                    success = generate_and_save_average_cycle_plot(
                        data=combined_data_filtered, 
                        r_peaks_indices=r_peaks_indices, 
                        fs=SAMPLING_RATE,
                        pre_r_ms=PRE_R_PEAK_MS, 
                        post_r_ms=POST_R_PEAK_MS, 
                        output_dir=current_output_subfolder, 
                        base_filename=output_filename_base_for_plot,
                        identifier=f"融合信号 ({txt_filename})"
                    )
                    if success:
                        saved_avg_cycle_images_count += 1

        print(f"\n--- 批量处理结束 ---")
        print(f"总共处理了 {processed_files_count} 个文件。")
        print(f"成功保存了 {saved_avg_cycle_images_count} 个平均心跳周期图片。")

