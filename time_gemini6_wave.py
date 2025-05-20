"""
批量心磁数据平均周期提取与图片保存脚本（小波变换R峰检测）
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import os
import re
import matplotlib.pyplot as plt
import matplotlib # 确保matplotlib被导入
import pywt # 导入PyWavelets

# --- 配置参数 (用户可根据需要调整) ---
SAMPLING_RATE = 1000  # Hz

# *** 指向包含所有每日数据文件夹的根目录 ***
ROOT_DATA_DIR = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg'
OUTPUT_FIGURE_DIR = r'C:\Users\Xiaoning Tan\Desktop\output_time' # *** 平均周期图片保存目录 ***

SKIP_HEADER_LINES = 3 
FILE_ENCODING = 'utf-8' 

# 滤波参数 (通用)
FILTER_ORDER_BANDPASS = 3 
LOWCUT_FREQ = 0.5         
HIGHCUT_FREQ = 45.0       
NOTCH_FREQ_MAINS = 50.0   
Q_FACTOR_NOTCH = 30.0     

# R峰检测参数 (通用 - 作为默认值)
DEFAULT_R_PEAK_MIN_HEIGHT_FACTOR = 0.3 
DEFAULT_R_PEAK_MIN_DISTANCE_MS = 150   

# 周期提取和平均参数 (通用)
PRE_R_PEAK_MS = 100       
POST_R_PEAK_MS = 150      

# --- 小波变换特定参数 (默认值) ---
DEFAULT_MOTHER_WAVELET = 'gaus1'  # 常用小波如 'gaus1', 'mexh', 'morl'
DEFAULT_WAVELET_SCALES = np.arange(8, 30) # 分析的尺度范围，需要根据信号特性和采样率调整

# --- 日龄特定的R峰检测参数 ---
# 现在可以包含 'mother_wavelet' 和 'wavelet_scales'
DAY_SPECIFIC_R_PEAK_PARAMS = {
    # 示例: Day 2-12
    "Day 2": {"height_factor": 0.25, "distance_ms": 220, "mother_wavelet": "gaus1", "wavelet_scales": np.arange(10, 35)},
    "Day 3": {"height_factor": 0.25, "distance_ms": 220}, # 将使用默认小波参数
    # ... (其他日龄配置) ...
    "Day 13": {"height_factor": 0.3, "distance_ms": 160},
    "Day 20": {"height_factor": 0.4, "distance_ms": 120, "mother_wavelet": "mexh", "wavelet_scales": np.arange(5, 25)},
    "Day 21": {"height_factor": 0.4, "distance_ms": 120},
}


# --- 函数定义 ---

def extract_day_label_from_folder(folder_name):
    match_detailed = re.search(r'[Dd](\d+)', folder_name) 
    if match_detailed:
        return f"Day {match_detailed.group(1)}"
    match_simple_day = re.search(r'day(\d+)', folder_name, re.IGNORECASE)
    if match_simple_day:
        return f"Day {match_simple_day.group(1)}"
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

def find_r_peaks_wavelet(data, fs, min_height_factor, min_distance_ms, 
                         mother_wavelet, scales, identifier="信号"):
    """
    使用小波变换检测R峰。
    """
    if data is None or len(data) == 0:
        return np.array([])

    # 1. 执行连续小波变换 (CWT)
    try:
        coefficients, frequencies = pywt.cwt(data, scales, mother_wavelet, sampling_period=1.0/fs)
    except Exception as e:
        print(f"  错误 ({identifier}): CWT 执行失败 - {e}")
        return np.array([])

    # 2. 处理小波系数 (例如，对尺度的绝对值求和)
    #    这是一种简化方法，更复杂的方法可能涉及分析特定尺度的模极大值线
    processed_coeffs = np.sum(np.abs(coefficients), axis=0)

    # 3. 在处理后的小波系数上应用峰值检测逻辑
    data_max_coeffs = np.max(processed_coeffs)
    if data_max_coeffs <= 1e-9: # 如果系数非常小
        robust_max_coeffs = np.percentile(processed_coeffs, 99)
        if robust_max_coeffs <= 1e-9:
            # print(f"  警告 ({identifier}): 处理后的小波系数最大值过小。")
            return np.array([])
        min_h_coeffs = robust_max_coeffs * min_height_factor
    else:
        min_h_coeffs = data_max_coeffs * min_height_factor

    if min_h_coeffs <= 1e-9:
        coeffs_std = np.std(processed_coeffs)
        min_h_coeffs = coeffs_std * 0.5 if coeffs_std > 1e-9 else 1e-6 # 使用更小的默认值，因为系数幅度可能不同
        # print(f"  警告 ({identifier}): 小波系数的计算高度阈值过低，已调整为: {min_h_coeffs:.3e}")


    min_dist_samples = int((min_distance_ms / 1000.0) * fs)
    height_param_coeffs = min_h_coeffs if min_h_coeffs > 1e-9 else None

    # print(f"  小波R峰检测 ({identifier}): height_factor={min_height_factor:.2f} (应用于系数最大值 {data_max_coeffs:.2e}), distance_ms={min_distance_ms}")
    # print(f"    calculated_min_h_coeffs={min_h_coeffs:.2e}, distance_samples={min_dist_samples}")


    try:
        peaks_indices, _ = find_peaks(processed_coeffs, height=height_param_coeffs, distance=min_dist_samples)
    except Exception as e:
        print(f"  错误 ({identifier}): 在小波系数上调用 find_peaks 时出错: {e}")
        return np.array([])
    
    # if len(peaks_indices) == 0:
        # print(f"  警告 ({identifier}): 使用小波方法未检测到R峰。")
    # else:
        # print(f"  ({identifier}): 使用小波方法检测到 {len(peaks_indices)} 个R峰。")
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
    averaged_cycle = np.mean(cycles_array, axis=0)
    std_dev_cycle = np.std(cycles_array, axis=0)
    cycle_time_axis_centered_at_r = np.linspace(-pre_r_ms / 1000.0, (post_r_samples -1) / 1000.0, total_cycle_samples)

    fig, ax = plt.subplots(figsize=(10, 6)) 
    plot_title = f'Averaged Cardiac Cycle: {base_filename}\n(Based on {valid_cycles_count} cycles, R-peak at 0s)'
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Time relative to R-peak (s)', fontsize=12)
    ax.set_ylabel('Signal Magnitude (pT)', fontsize=12)
    ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='grey')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.4', color='lightgrey')
    ax.minorticks_on()

    num_bg_cycles_to_plot = min(len(cycles_array), 20) 
    indices_to_plot = np.random.choice(len(cycles_array), num_bg_cycles_to_plot, replace=False) if len(cycles_array) > num_bg_cycles_to_plot else np.arange(len(cycles_array))
    for idx in indices_to_plot:
        ax.plot(cycle_time_axis_centered_at_r, cycles_array[idx, :], color='lightgray', alpha=0.35, linewidth=0.7)

    ax.plot(cycle_time_axis_centered_at_r, averaged_cycle, color='orangered', linewidth=2, label='Averaged Cycle')
    ax.fill_between(cycle_time_axis_centered_at_r,
                       averaged_cycle - std_dev_cycle,
                       averaged_cycle + std_dev_cycle,
                       color='lightcoral', alpha=0.35, label='Avg ± 1 Std Dev')
    ax.legend(loc='best')
    plt.tight_layout()

    try:
        output_image_filename = f"{base_filename}_avg_cycle_wavelet.png" # 添加wavelet标识
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

        for day_folder_name in sorted(os.listdir(ROOT_DATA_DIR)):
            day_folder_path = os.path.join(ROOT_DATA_DIR, day_folder_name)
            
            if os.path.isdir(day_folder_path):
                day_label = extract_day_label_from_folder(day_folder_name) 
                print(f"\n--- 开始处理文件夹: {day_folder_name} (日龄标签: {day_label}) ---")
                
                # 获取当前日龄的特定R峰检测参数 (包括小波参数)
                day_r_peak_config = DAY_SPECIFIC_R_PEAK_PARAMS.get(day_label, {})
                current_r_peak_height_factor = day_r_peak_config.get("height_factor", DEFAULT_R_PEAK_MIN_HEIGHT_FACTOR)
                current_r_peak_distance_ms = day_r_peak_config.get("distance_ms", DEFAULT_R_PEAK_MIN_DISTANCE_MS)
                current_mother_wavelet = day_r_peak_config.get("mother_wavelet", DEFAULT_MOTHER_WAVELET)
                current_wavelet_scales = day_r_peak_config.get("wavelet_scales", DEFAULT_WAVELET_SCALES)
                
                print(f"  使用R峰检测参数: Height Factor = {current_r_peak_height_factor:.2f}, Distance MS = {current_r_peak_distance_ms}")
                print(f"  使用小波参数: Mother Wavelet = {current_mother_wavelet}, Scales = {current_wavelet_scales[0]}...{current_wavelet_scales[-1]}")

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
                    
                    # 使用新的小波R峰检测函数
                    r_peaks_indices = find_r_peaks_wavelet(
                        data=combined_data_filtered, 
                        fs=SAMPLING_RATE,
                        min_height_factor=current_r_peak_height_factor,
                        min_distance_ms=current_r_peak_distance_ms,
                        mother_wavelet=current_mother_wavelet,
                        scales=current_wavelet_scales,
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
            # else: 
                # print(f"跳过项目 (非文件夹): {day_folder_name}")

        print(f"\n--- 批量处理结束 ---")
        print(f"总共处理了 {processed_files_count} 个文件。")
        print(f"成功保存了 {saved_avg_cycle_images_count} 个平均心跳周期图片。")

