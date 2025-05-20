import numpy as np
import zhplot
import matplotlib.pyplot as plt
from scipy import signal

# 生物磁信号处理器类
class BiomagProcessor:
    def __init__(self, fs=1000):
        self.fs = fs
        
        # 设计50Hz陷波滤波器
        self.notch_Q = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(50.0, self.notch_Q, self.fs)
        
        # 设计0.5Hz高通滤波器
        self.highpass_cutoff = 0.5
        self.sos_highpass = signal.butter(8, self.highpass_cutoff, 'hp', fs=self.fs, output='sos')

    def process_pipeline(self, data):
        """三级处理流水线"""
        # 阶段1：动态基线修正
        baseline = self._adaptive_baseline_correction(data)
        processed = data - baseline
        
        # 阶段2：零相移高通滤波
        processed = signal.sosfiltfilt(self.sos_highpass, processed)
        
        # 阶段3：50Hz陷波滤波
        processed = signal.filtfilt(self.b_notch, self.a_notch, processed)
        return processed, baseline

    def _adaptive_baseline_correction(self, data):
        """混合基线追踪算法"""
        window_size = int(10 * self.fs) | 1  # 10秒窗口
        med_filtered = signal.medfilt(data, kernel_size=window_size)
        
        # 二次多项式拟合
        coeffs = np.polyfit(np.arange(len(data)), med_filtered, 2)
        return np.polyval(coeffs, np.arange(len(data)))

# 主程序
if __name__ == "__main__":
    # 数据加载（注意修改文件路径）
    data = np.loadtxt(r'C:\Users\Administrator\Desktop\朱鹮_250426\朱鹮day22_t1.txt', skiprows=2, encoding="utf-8")
    Bx_raw, By_raw = data[:, 0], data[:, 1]
    
    # 初始化处理器
    processor = BiomagProcessor(fs=1000)
    t = np.arange(len(Bx_raw)) / 1000.0  # 时间序列
    
    # 信号处理
    Bx_processed, _ = processor.process_pipeline(Bx_raw)
    By_processed, _ = processor.process_pipeline(By_raw)

    # 创建对比画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    time_slice = slice(35000, 40000)  # 35-40秒区间

    # 绘制Bx通道对比
    ax1.plot(t[time_slice], Bx_raw[time_slice], 'lightblue', lw=0.8, label='原始信号')
    ax1.plot(t[time_slice], Bx_processed[time_slice], 'darkblue', lw=1.2, label='处理后')
    ax1.set(xlabel='时间 (s)', ylabel='磁场强度 (pT)', 
          title='Bx通道 QRS复合波对比')
    ax1.legend(fontsize=7, frameon=False)

    # 绘制By通道对比
    ax2.plot(t[time_slice], By_raw[time_slice], 'lightgreen', lw=0.8, label='原始信号')
    ax2.plot(t[time_slice], By_processed[time_slice], 'darkgreen', lw=1.2, label='处理后')
    ax2.set(xlabel='时间 (s)', title='By通道 QRS复合波对比')
    ax2.legend(fontsize=7, frameon=False)

    # 样式统一设置
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, ls=':')
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_xlim(35, 40)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
