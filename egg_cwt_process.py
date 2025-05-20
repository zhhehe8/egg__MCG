import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


## 该脚本用于批量处理文件夹下的所有.txt文件，进行时频分析，并保存为.png格式的图片。

def process_file(input_file, output_dir):
    """处理单个文件的核心函数"""
    try:
        # 1. 数据加载
        # 加载空载数据"20250218_空载1.txt"
        empty_data = np.loadtxt(r'C:\Users\Xiaoning Tan\Desktop\egg_2025\空载250218\20250218_空载1.txt', skiprows=2, encoding="utf-8")
        empty_Bx = empty_data[:, 0]  # 第一列数据Bx
        empty_By = empty_data[:, 1]  # 第二列数据By
        
        # 读取文件夹下的所有.txt文件
        data = np.loadtxt(input_file, skiprows=2, encoding="utf-8")
        col1 = data[:, 0]
        col2 = data[:, 1]
        fs = 1000

        # 2. 创建画布,2*2子图
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        ax1, ax2 = axs[0, 0], axs[0, 1]
        ax3, ax4 = axs[1, 0], axs[1, 1]

        # 3. 处理第一列数据Bx
        f1, t1, Zxx1 = signal.stft(col1, fs, nperseg=1024, noverlap=512)
        mask = (f1 <= 40)  # 筛选0-40Hz
        im1 = ax1.pcolormesh(t1, f1[mask], np.abs(Zxx1[mask]), 
                        shading='gouraud', cmap='bwr')
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_title('Bx Time-Frequency Analysis')
        fig.colorbar(im1, ax=ax1, label='Magnitude')
        # 设置colorbar范围为0-0.25Hz
        im1.set_clim(0, 0.25)
        # 设置x轴范围为0-60s
        ax1.set_xlim(0, 60)
        # 设置y轴范围为0.5-8Hz
        ax1.set_ylim(0.5, 8)
        # 3.1 处理空载的第一列数据Bx
        f3, t3, Zxx3 = signal.stft(empty_Bx, fs, nperseg=1024, noverlap=512)
        mask = (f3 <= 40)  # 筛选0-40Hz
        im3 = ax3.pcolormesh(t3, f3[mask], np.abs(Zxx3[mask]), 
                        shading='gouraud', cmap='bwr')
        ax3.set_ylabel('Frequency [Hz]')
        ax3.set_title('Empty Bx Time-Frequency Analysis')
        fig.colorbar(im3, ax=ax3, label='Magnitude')
        # 设置colorbar范围为0-1Hz
        im3.set_clim(0, 0.25)
        # 设置y轴范围为1-6Hz
        ax3.set_ylim(0.5, 8)

        ax3.set_xlim(0, 60)

        # 4. 处理第二列数据
        f2, t2, Zxx2 = signal.stft(col2, fs, nperseg=1024, noverlap=512)
        mask = (f2 <= 40)  # 筛选0-40Hz
        im2 = ax2.pcolormesh(t2, f2[mask], np.abs(Zxx2[mask]),
                    shading='gouraud', cmap='bwr')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_title('By Time-Frequency Analysis')
        fig.colorbar(im2, ax=ax2, label='Magnitude')
        # 设置colorbar范围为0-0.25Hz
        im2.set_clim(0, 0.25)
        # 设置y轴范围为1-6Hz
        ax2.set_ylim(0.5, 8)

        ax2.set_xlim(0, 60)
        # 4.1 处理空载的第二列数据By
        f4, t4, Zxx4 = signal.stft(empty_By, fs, nperseg=1024, noverlap=512)
        mask = (f4 <= 40)  # 筛选0-40Hz
        im4 = ax4.pcolormesh(t4, f4[mask], np.abs(Zxx4[mask]), 
                        shading='gouraud', cmap='bwr')
        ax4.set_ylabel('Frequency [Hz]')
        ax4.set_title('Empty By Time-Frequency Analysis')
        fig.colorbar(im4, ax=ax4, label='Magnitude')
        im4.set_clim(0, 0.25)
        ax4.set_ylim(0.5, 8)
        ax4.set_xlim(0, 60)

        # 5. 保存输出
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"成功处理：{input_file} -> {output_path}")
    except Exception as e:
        print(f"处理失败：{input_file}，错误：{str(e)}")

def batch_process(input_folder, output_folder):
    """批量处理函数"""
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入目录[3](@ref)[6](@ref)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                process_file(file_path, output_folder)

if __name__ == "__main__":
    # 配置路径
    input_folder = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d19'  # 原始数据目录
    output_folder = r'C:\Users\Xiaoning Tan\Desktop\output\day19'  # 输出图片目录

    # 执行批量处理
    batch_process(input_folder, output_folder)
    print("批量处理完成！")
