"""
# 处理小鸡心电图数据，使用Pan-Tompkins算法提取QRS波形并计算平均QRS波形
"""
import numpy as np
import zhplot
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, iirnotch # 仅用于可选的陷波滤波
from biosppy.signals import ecg # biosppy用于ECG处理
import os

# --- 配置参数 (用户可根据需要调整) ---
SAMPLING_RATE = 1000  # Hz, 采样率
# 请将FILE_PATH替换为您的实际文件路径
FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t3.txt' 
SKIP_HEADER_LINES = 3 # 数据文件头部需要跳过的行数
FILE_ENCODING = 'utf-8' # 数据文件编码
OUTPUT_FIGURE_DIR = r'C:\Users\Xiaoning Tan\Desktop\egg_figure_qrs' # 图片保存目录

# QRS波形提取窗口 (围绕R峰) - 需要根据小鸡ECG特性调整
QRS_PRE_R_MS = 60   # ms, R峰前提取时间 (例如Q波开始前)
QRS_POST_R_MS = 80  # ms, R峰后提取时间 (例如S波结束后)

# 可选：陷波滤波参数 (如果存在50/60Hz工频干扰)
APPLY_NOTCH_FILTER = True # 是否应用陷波滤波器
NOTCH_FREQ_MAINS = 50.0   # Hz, 工频干扰频率
Q_FACTOR_NOTCH = 30.0     # 陷波滤波器的品质因数

# --- 中文字体设置 (如果需要，确保系统中有对应字体) ---
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体，例如黑体
# plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# --- 函数定义 ---

def load_cardiac_data(filepath, skip_header, file_encoding='utf-8'):
    """
    从文本文件加载心磁数据。预期至少有两列。
    返回: channel1_data, channel2_data
    """
    try:
        data = np.loadtxt(filepath, skiprows=skip_header, dtype=float, encoding=file_encoding)
        if data.ndim == 1:
            print(f"警告: 文件 {os.path.basename(filepath)} 只包含一列数据。")
            if data.shape[0] > 1: # 确保不是只有一个数字
                 return data, None # 将单列数据作为channel1返回
            else:
                print(f"错误：数据文件 {os.path.basename(filepath)} 数据点过少。")
                return None, None
        if data.shape[1] < 1: # 至少需要一列
            print(f"错误：数据文件 {os.path.basename(filepath)} 列数不足。")
            return None, None
        
        channel1 = data[:, 0]
        channel2 = data[:, 1] if data.shape[1] >= 2 else None # 如果只有一列，channel2为None
        
        print(f"成功加载文件: {os.path.basename(filepath)}, 通道1点数: {len(channel1)}")
        if channel2 is not None:
            print(f"通道2点数: {len(channel2)}")
        return channel1, channel2
    except Exception as e:
        print(f"读取文件 '{os.path.basename(filepath)}' 或解析数据时出错: {e}")
        return None, None

def apply_custom_notch_filter(data, notch_freq, quality_factor, fs):
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
    filtered_data = filtfilt(b, a, data) # 使用filtfilt进行零相位滤波
    return filtered_data

# --- 主程序 ---
if __name__ == "__main__":
    print(f"开始处理文件: {FILE_PATH}")
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True) # 确保输出目录存在
    file_base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]

    # 1. 加载数据
    ch1_data_raw, ch2_data_raw = load_cardiac_data(FILE_PATH, SKIP_HEADER_LINES, FILE_ENCODING)

    if ch1_data_raw is None:
        print("错误: 未能加载通道1的数据。程序终止。")
    else:
        # 选择一个通道进行处理，这里默认使用通道1 (Bx)
        # Pan-Tompkins通常应用于单导联ECG信号
        signal_to_process = ch1_data_raw
        channel_label = "通道1 (Bx)" # 或 "通道2 (By)" 如果选择ch2_data_raw

        print(f"\n--- 开始处理 {channel_label} ---")

        # 2. 可选：应用自定义陷波滤波器 (biosppy的ecg函数内部也会进行滤波)
        if APPLY_NOTCH_FILTER:
            print(f"应用自定义 {NOTCH_FREQ_MAINS}Hz 陷波滤波器...")
            signal_to_process = apply_custom_notch_filter(signal_to_process, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
            if signal_to_process is None:
                print("错误: 陷波滤波失败。")
        
        # 3. 使用 biosppy.signals.ecg.ecg 进行处理
        #    该函数内部会进行滤波和R峰检测 (例如使用Hamilton检测器，其原理借鉴了Pan-Tompkins)
        print("使用 biosppy.signals.ecg.ecg 进行R峰检测...")
        try:
            # `ecg()` 函数返回一个包含多个键的 Biosignal 对象 (类似字典)
            # 主要关注: 'ts' (时间轴), 'filtered' (滤波后的信号), 'rpeaks' (R峰索引)
            ecg_results = ecg.ecg(signal=signal_to_process, sampling_rate=SAMPLING_RATE, show=False)
        except Exception as e:
            print(f"错误: biosppy ECG处理失败 - {e}")
            ecg_results = None # 或者 exit()

        if ecg_results:
            filtered_signal = ecg_results['filtered']
            rpeaks_indices = ecg_results['rpeaks']
            ts_ecg = ecg_results['ts'] # biosppy生成的时间轴

            if len(rpeaks_indices) == 0:
                print("警告: biosppy 未能检测到任何R峰。请检查信号质量或调整参数。")
            else:
                print(f"biosppy 检测到 {len(rpeaks_indices)} 个R峰。")

                # 4. 绘制一段滤波后信号和检测到的R峰
                plt.figure(figsize=(15, 5))
                # 只绘制前10秒或整个信号（如果较短）
                plot_duration_samples = min(len(filtered_signal), SAMPLING_RATE * 10) 
                plt.plot(ts_ecg[:plot_duration_samples], filtered_signal[:plot_duration_samples], label="滤波后信号 (biosppy)")
                
                # 仅绘制在显示范围内的R峰
                rpeaks_in_plot_range = rpeaks_indices[rpeaks_indices < plot_duration_samples]
                plt.plot(ts_ecg[rpeaks_in_plot_range], filtered_signal[rpeaks_in_plot_range], 'rx', markersize=8, label="检测到的R峰")
                
                plt.title(f"{channel_label} - 滤波后信号与R峰检测 (前{plot_duration_samples/SAMPLING_RATE:.1f}秒)")
                plt.xlabel("时间 (秒)")
                plt.ylabel("幅度")
                plt.legend()
                plt.grid(True)
                
                # 保存R峰检测图
                rpeaks_fig_path = os.path.join(OUTPUT_FIGURE_DIR, f"{file_base_name}_{channel_label.replace(' ', '_')}_Rpeaks_biosppy.png")
                try:
                    plt.savefig(rpeaks_fig_path, dpi=300)
                    print(f"R峰检测图已保存到: {rpeaks_fig_path}")
                except Exception as e:
                    print(f"错误：无法保存R峰检测图。原因: {e}")
                plt.show()
                plt.close()


                # 5. 提取QRS波形并计算平均QRS波形
                qrs_pre_samples = int((QRS_PRE_R_MS / 1000.0) * SAMPLING_RATE)
                qrs_post_samples = int((QRS_POST_R_MS / 1000.0) * SAMPLING_RATE)
                total_qrs_samples = qrs_pre_samples + qrs_post_samples

                all_qrs_complexes = []
                valid_qrs_count = 0

                if total_qrs_samples <=0:
                    print("错误：QRS提取窗口长度无效。")
                else:
                    for r_idx in rpeaks_indices:
                        start_idx = r_idx - qrs_pre_samples
                        end_idx = r_idx + qrs_post_samples

                        if start_idx >= 0 and end_idx < len(filtered_signal):
                            qrs_segment = filtered_signal[start_idx:end_idx]
                            if len(qrs_segment) == total_qrs_samples: # 确保长度一致
                                all_qrs_complexes.append(qrs_segment)
                                valid_qrs_count +=1
                    
                    if valid_qrs_count > 1: # 至少需要两个QRS波才能平均
                        qrs_array = np.array(all_qrs_complexes)
                        averaged_qrs = np.mean(qrs_array, axis=0)
                        std_dev_qrs = np.std(qrs_array, axis=0) # 可选：计算标准差
                        
                        # 创建QRS波形的时间轴 (相对于R峰)
                        qrs_time_axis = np.linspace(-QRS_PRE_R_MS / 1000.0, 
                                                    (qrs_post_samples -1) / 1000.0, # 确保时间轴点数正确
                                                    total_qrs_samples)


                        print(f"成功提取并平均了 {valid_qrs_count} 个QRS波形。")

                        # 6. 绘制平均QRS波形
                        plt.figure(figsize=(8, 6))
                        plt.title(f"{channel_label} - 平均QRS波形 (基于 {valid_qrs_count} 个周期)")
                        plt.xlabel("相对于R峰的时间 (秒)")
                        plt.ylabel("幅度")
                        
                        # 绘制背景中的一些单个QRS波形
                        num_bg_cycles_to_plot = min(len(qrs_array), 20) 
                        indices_to_plot = np.random.choice(len(qrs_array), num_bg_cycles_to_plot, replace=False) if len(qrs_array) > num_bg_cycles_to_plot else np.arange(len(qrs_array))
                        for i_plot in indices_to_plot:
                            plt.plot(qrs_time_axis, qrs_array[i_plot, :], color='lightgray', alpha=0.5, linewidth=0.8)
                        
                        plt.plot(qrs_time_axis, averaged_qrs, color='red', linewidth=2, label="平均QRS波形")
                        plt.fill_between(qrs_time_axis, 
                                           averaged_qrs - std_dev_qrs, 
                                           averaged_qrs + std_dev_qrs, 
                                           color='lightcoral', alpha=0.3, label="平均值 ± 1 标准差")
                        plt.axvline(0, color='grey', linestyle='--', linewidth=0.8, label="R峰位置 (0s)") # 标记R峰在0s
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()

                        # 保存平均QRS波形图
                        avg_qrs_fig_path = os.path.join(OUTPUT_FIGURE_DIR, f"{file_base_name}_{channel_label.replace(' ', '_')}_averaged_QRS.png")
                        try:
                            plt.savefig(avg_qrs_fig_path, dpi=300)
                            print(f"平均QRS波形图已保存到: {avg_qrs_fig_path}")
                        except Exception as e:
                            print(f"错误：无法保存平均QRS波形图。原因: {e}")
                        plt.show()
                        plt.close()

                    else:
                        print("警告: 提取到的有效QRS波形数量不足 (<2)，无法进行平均。")
        else:
            print("biosppy未能成功处理信号。")


    print("\n--- 处理结束 ---")
