import numpy as np
import zhplot
import matplotlib.pyplot as plt
from scipy import signal

# 增强型心磁信号处理流水线
class BiomagProcessor:
    def __init__(self, fs=1000):
        self.fs = fs
        
        # 设计50Hz陷波滤波器
        self.notch_freq = 50.0
        self.notch_Q = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(
            self.notch_freq, self.notch_Q, self.fs
        )
        
        # 设计0.5Hz高通滤波器
        self.highpass_cutoff = 0.5
        self.sos_highpass = signal.butter(
            8, self.highpass_cutoff, 'hp', fs=self.fs, output='sos'
        )

    def process_pipeline(self, data):
        """三级处理流程：基线修正 → 高通滤波 → 工频陷波"""
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
        # 滑动窗口中值滤波（10秒窗口）
        window_size = int(10 * self.fs) | 1  # 强制为奇数
        med_filtered = signal.medfilt(data, kernel_size=window_size)
        
        # 二次多项式拟合
        coeffs = np.polyfit(np.arange(len(data)), med_filtered, deg=2)
        baseline = np.polyval(coeffs, np.arange(len(data)))
        return baseline

# 可视化配置
def plot_compare(ax, t, raw, processed, title, ylim=(-1, 1)):
    ax.plot(t, raw, 'lightgray', label='Raw')
    ax.plot(t, processed, 'navy', linewidth=1, label='Processed')
    ax.set_xlim(0, 40)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=6)

# 主程序 ----------------------------
if __name__ == "__main__":
    # 数据加载
    data = np.loadtxt(r'C:\Users\Administrator\Desktop\朱鹮_250426\朱鹮day22_t1.txt', skiprows=2, encoding="utf-8")
    Bx_raw, By_raw = data[:, 0], data[:, 1]
    
    empty_data = np.loadtxt(r'C:\Users\Administrator\Desktop\朱鹮_250426\朱鹮未受精蛋1_t2.txt', skiprows=2, encoding="utf-8")
    empty_Bx, empty_By = empty_data[:, 0], empty_data[:, 1]

    # 初始化处理器
    processor = BiomagProcessor(fs=1000)
    t = np.arange(len(Bx_raw)) / 1000.0  # 时间轴生成

    # 处理主要数据
    Bx_processed, Bx_baseline = processor.process_pipeline(Bx_raw)
    By_processed, By_baseline = processor.process_pipeline(By_raw)

    # 处理背景噪声数据
    empty_Bx_processed, _ = processor.process_pipeline(empty_Bx)
    empty_By_processed, _ = processor.process_pipeline(empty_By)

    # 可视化全流程对比
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2)

    # 基线修正展示
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, Bx_raw, 'steelblue', label='Bx原始')
    ax0.plot(t, Bx_baseline, 'darkred', label='基线估计')
    ax0.set_xlim(0,40), ax0.legend()
    ax0.set_title("基线漂移估计 (Bx通道)")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, By_raw, 'forestgreen', label='By原始')
    ax1.plot(t, By_baseline, 'darkred', label='基线估计')
    ax1.set_xlim(0,40), ax1.legend()
    ax1.set_title("基线漂移估计 (By通道)")

    # 处理流程对比
    plot_compare(fig.add_subplot(gs[1, 0]), t, Bx_raw, Bx_processed, "Bx处理全流程")
    plot_compare(fig.add_subplot(gs[1, 1]), t, By_raw, By_processed, "By处理全流程")

    # 细节放大对比
    detail_ax1 = fig.add_subplot(gs[2, 0])
    detail_ax1.plot(t[35000:40000], Bx_raw[35000:40000], 'lightblue', label='原始')
    detail_ax1.plot(t[35000:40000], Bx_processed[35000:40000], 'darkblue', label='处理后')
    detail_ax1.set_title("Bx通道QRS复合波放大（35-40秒）")

    detail_ax2 = fig.add_subplot(gs[2, 1])
    detail_ax2.plot(t[35000:40000], By_raw[35000:40000], 'lightgreen')
    detail_ax2.plot(t[35000:40000], By_processed[35000:40000], 'darkgreen')
    detail_ax2.set_title("By通道QRS复合波放大（35-40秒）")

    plt.tight_layout()
    plt.show()
