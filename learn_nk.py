import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import os

# --- 配置参数 (用户可根据需要调整) ---
SAMPLING_RATE = 1000  # Hz, 采样率
# 请将FILE_PATH替换为您的实际文件路径
FILE_PATH = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t1_待破壳.txt' 
SKIP_HEADER_LINES = 2 # 数据文件头部需要跳过的行数
FILE_ENCODING = 'utf-8' # 数据文件编码
OUTPUT_FIGURE_DIR = r'.' # 图片保存目录 (当前目录)

# 选择要处理的数据列 (0代表第一列Bx, 1代表第二列By)
# NeuroKit2的ECG处理通常针对单导联信号
CHANNEL_TO_PROCESS = 0 # 默认处理第一列 (Bx)

# --- 中文字体设置 (可选, 如果绘图时中文显示有问题) ---
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 例如: SimHei (黑体)
# plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# --- 函数定义 ---

def load_cardiac_data_nk(filepath, skip_header, file_encoding='utf-8', channel_index=0):
    """
    从文本文件加载心磁数据，并选择指定通道。
    返回: 单通道的numpy数组
    """
    try:
        # 使用pandas更灵活地处理可能不规则的列数或数据类型问题
        df = pd.read_csv(filepath, skiprows=skip_header, delim_whitespace=True, encoding=file_encoding, header=None, on_bad_lines='skip')
        
        if df.shape[1] <= channel_index:
            print(f"  错误：文件 {os.path.basename(filepath)} 的列数少于指定的通道索引 {channel_index}。")
            return None
            
        # 尝试将选定的列转换为数值型，非数值转为NaN
        selected_channel_data = pd.to_numeric(df.iloc[:, channel_index], errors='coerce')
        
        # 移除NaN值 (如果存在)
        selected_channel_data = selected_channel_data.dropna().to_numpy()

        if selected_channel_data.size == 0:
            print(f"  错误：在文件 {os.path.basename(filepath)} 的通道 {channel_index} 中未找到有效数值数据。")
            return None
            
        print(f"成功加载文件: {os.path.basename(filepath)}, 处理通道: {channel_index}, 数据点数: {len(selected_channel_data)}")
        return selected_channel_data
    except Exception as e:
        print(f"  读取文件 '{os.path.basename(filepath)}' 或解析数据时出错: {e}")
        return None

# --- 主程序 ---
if __name__ == "__main__":
    print(f"开始处理文件: {FILE_PATH}")
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True) 
    file_base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]

    # 1. 加载数据
    ecg_signal = load_cardiac_data_nk(FILE_PATH, SKIP_HEADER_LINES, FILE_ENCODING, channel_index=CHANNEL_TO_PROCESS)

    if ecg_signal is None:
        print("错误: 未能加载心磁数据。程序终止。")
    else:
        print(f"\n--- 开始使用 NeuroKit2 处理通道 {CHANNEL_TO_PROCESS} ---")
        
        try:
            signals, info = nk.ecg_process(ecg_signal, sampling_rate=SAMPLING_RATE, method='neurokit')
            rpeaks_indices = info["ECG_R_Peaks"]
            
            if len(rpeaks_indices) == 0:
                print("警告: NeuroKit2 未能检测到任何R峰。请检查信号质量或尝试不同的处理方法/参数。")
            else:
                print(f"NeuroKit2 检测到 {len(rpeaks_indices)} 个R峰。")

                # 3. 可视化处理结果
                # NeuroKit2 提供了便捷的绘图功能
                # *** 修改点：移除了 ecg_plot 中的 sampling_rate 参数 ***
                nk.ecg_plot(signals) 
                plt.suptitle(f"NeuroKit2 ECG Processing: {file_base_name} - Channel {CHANNEL_TO_PROCESS}", y=1.02)
                
                figure_save_path = os.path.join(OUTPUT_FIGURE_DIR, f"{file_base_name}_channel{CHANNEL_TO_PROCESS}_neurokit_processing.png")
                try:
                    plt.savefig(figure_save_path, dpi=300, bbox_inches='tight')
                    print(f"NeuroKit2 处理结果图已保存到: {figure_save_path}")
                except Exception as e:
                    print(f"错误：无法保存 NeuroKit2 处理结果图。原因: {e}")
                plt.show()
                plt.close()

                # 4. 提取和平均QRS波群
                qrs_pre_ms = 60  
                qrs_post_ms = 80 
                
                qrs_pre_samples = int(qrs_pre_ms * SAMPLING_RATE / 1000)
                qrs_post_samples = int(qrs_post_ms * SAMPLING_RATE / 1000)
                
                # 使用 'ECG_Clean' 进行周期提取
                epochs = nk.epochs_create(signals['ECG_Clean'], 
                                          events=rpeaks_indices, 
                                          sampling_rate=SAMPLING_RATE, 
                                          epochs_start=-qrs_pre_samples/SAMPLING_RATE, 
                                          epochs_end=qrs_post_samples/SAMPLING_RATE)   
                
                all_qrs_segments = []
                for epoch_key in epochs:
                    segment = epochs[epoch_key]['Signal'].values
                    # 确保所有片段长度一致
                    # 对于NeuroKit2的epochs_create，它通常会确保长度一致，但检查一下无妨
                    # 期望的长度是 pre_samples + post_samples (因为R峰本身算一个点，所以窗口是-pre到+post-1)
                    # 或者更准确地说是 total_duration_samples = qrs_pre_samples + qrs_post_samples
                    expected_len = qrs_pre_samples + qrs_post_samples
                    if len(segment) == expected_len:
                        all_qrs_segments.append(segment)
                    # else: # 调试时可以取消注释
                        # print(f"  警告: 段 {epoch_key} 长度 {len(segment)} 与期望长度 {expected_len} 不符。")

                
                if len(all_qrs_segments) > 1:
                    averaged_qrs = np.mean(np.array(all_qrs_segments), axis=0)
                    # 创建时间轴，单位为毫秒，R峰在0ms
                    time_axis = np.linspace(-qrs_pre_ms, qrs_post_ms - (1000/SAMPLING_RATE), len(averaged_qrs), endpoint=False)


                    plt.figure(figsize=(8, 6))
                    plt.plot(time_axis, averaged_qrs, label="平均QRS波形 (NeuroKit)", color='red')
                    
                    for i in range(min(len(all_qrs_segments), 10)): 
                         plt.plot(time_axis, all_qrs_segments[i], color='grey', alpha=0.3)
                    plt.title(f"平均QRS波形 - {file_base_name} (基于 {len(all_qrs_segments)} 个周期)")
                    plt.xlabel("相对于R峰的时间 (ms)")
                    plt.ylabel("幅度 (pT)") # 假设原始单位是pT
                    plt.axvline(0, color='grey', linestyle='--', label="R峰 (0 ms)")
                    plt.legend()
                    plt.grid(True)
                    
                    avg_qrs_fig_path = os.path.join(OUTPUT_FIGURE_DIR, f"{file_base_name}_channel{CHANNEL_TO_PROCESS}_averaged_QRS_neurokit.png")
                    try:
                        plt.savefig(avg_qrs_fig_path, dpi=300, bbox_inches='tight')
                        print(f"平均QRS波形图已保存到: {avg_qrs_fig_path}")
                    except Exception as e:
                        print(f"错误：无法保存平均QRS波形图。原因: {e}")
                    plt.show()
                    plt.close()
                else:
                    print("提取到的QRS片段不足以进行平均。")

        except Exception as e:
            print(f"处理文件 {FILE_PATH} 时发生错误: {e}")

    print("\n--- 处理结束 ---")

