import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def process_file(input_file, output_dir, empty_file='20250218_空载1.txt'):
    """处理单个文件的核心函数
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        empty_file: 空载数据文件路径 ('20250218_空载1.txt')
    """
    try:
        # 1. 数据加载
        empty_data = np.loadtxt(empty_file, skiprows=2, encoding="utf-8")
        empty_Bx = empty_data[:, 0]
        empty_By = empty_data[:, 1]

        data = np.loadtxt(input_file, skiprows=2, encoding="utf-8")
        col1 = data[:, 0]
        col2 = data[:, 1]
        fs = 1000

        # 2. 创建画布
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        ax1, ax2 = axs[0, 0], axs[0, 1]
        ax3, ax4 = axs[1, 0], axs[1, 1]

        # 3. Bx,By处理
        def plot_spectrogram(ax, data, title, empty_data=None):
            f, t, Zxx = signal.stft(data, fs, nperseg=1024, noverlap=512)
            mask = (f <= 40)
            im = ax.pcolormesh(t, f[mask], np.abs(Zxx[mask]), 
                             shading='gouraud', cmap='viridis')
            ax.set_title(title)
            ax.set_ylim(1, 6)
            ax.set_xlim(0, 40)
            im.set_clim(0, 1)
            return im

        # 第一列处理
        im1 = plot_spectrogram(ax1, col1, 'Bx Time-Frequency Analysis')
        fig.colorbar(im1, ax=ax1, label='Magnitude')
        im3 = plot_spectrogram(ax3, empty_Bx, 'Empty Bx Analysis')
        fig.colorbar(im3, ax=ax3, label='Magnitude')

        # 第二列处理
        im2 = plot_spectrogram(ax2, col2, 'By Time-Frequency Analysis')
        fig.colorbar(im2, ax=ax2, label='Magnitude')
        im4 = plot_spectrogram(ax4, empty_By, 'Empty By Analysis')
        fig.colorbar(im4, ax=ax4, label='Magnitude')

        # 4. 设置公共标签
        for ax in axs.flat:
            if ax in [ax3, ax4]:
                ax.set_xlabel('Time [s]')
            if ax in [ax1, ax3]:
                ax.set_ylabel('Frequency [Hz]')

        # 5. 保存输出
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"成功处理：{input_file} -> {output_path}")
    except Exception as e:
        print(f"处理失败：{input_file}，错误：{str(e)}")

def batch_process(input_folder, output_folder):
    """批量处理函数
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                process_file(file_path, output_folder)
