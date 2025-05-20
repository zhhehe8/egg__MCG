import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# 1. 加载数据
data = np.loadtxt(r'C:\Users\Xiaoning Tan\Desktop\egg_2025\朱鹮_250426\朱鹮day25_2_t2.txt', skiprows=2, encoding="utf-8")
# data = np.loadtxt(r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t3.txt', skiprows=2, encoding="utf-8")
Bx = data[:, 0]
By = data[:, 1]
fs = 1000  # 采样率

# 加载空载数据
empty_data = np.loadtxt(r'C:\Users\Xiaoning Tan\Desktop\egg_2025\朱鹮_250426\空载1.txt', skiprows=2, encoding="utf-8")
# empty_data = np.loadtxt(r'C:\Users\Xiaoning Tan\Desktop\egg_2025\空载250218\20250218_空载1.txt', skiprows=2, encoding="utf-8")
empty_Bx = empty_data[:, 0]
empty_By = empty_data[:, 1]

# 2. 创建画布：3行2列
fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=False)   
ax0, ax5 = axs[0]         # 第一行：原始时域图（实验 & 空载）
ax1, ax2 = axs[1]         # 第二行：实验数据时频图
ax3, ax4 = axs[2]         # 第三行：空载数据时频图

# 3. 实验数据时域图（Bx & By）
t = np.arange(len(Bx)) / fs
ax0.plot(t, Bx, label='Bx', color='blue', linewidth=1)
ax0.plot(t, By, label='By', color='red', linewidth=1, alpha=0.7)
ax0.set_title("Raw Time-Domain Signals (Experiment)")
# ax0.set_xlabel("Time [s]")
ax0.set_ylabel("Amplitude")
ax0.legend(loc="upper right")
ax0.set_xlim(0, 40)

# 4. 空载数据时域图（empty_Bx & empty_By）
t_empty = np.arange(len(empty_Bx)) / fs
ax5.plot(t_empty, empty_Bx, label='Empty Bx', color='blue', linewidth=1)
ax5.plot(t_empty, empty_By, label='Empty By', color='red', linewidth=1, alpha=0.7)
ax5.set_title("Raw Time-Domain Signals (Empty)")
# ax5.set_xlabel("Time [s]")
ax5.set_ylabel("Amplitude")
ax5.legend(loc="upper right")
ax5.set_xlim(0, 40)

# 5. 时频图 - 实验 Bx
f1, t1, Zxx1 = signal.stft(Bx, fs, nperseg=1024, noverlap=512)
mask = (f1 <= 40)
im1 = ax1.pcolormesh(t1, f1[mask], np.abs(Zxx1[mask]), shading='gouraud', cmap='bwr')
ax1.set_ylabel('Frequency [Hz]')
ax1.set_title('Bx Time-Frequency Analysis')
fig.colorbar(im1, ax=ax1, label='Magnitude')
im1.set_clim(0, 0.25)
ax1.set_ylim(1.5, 10)
ax1.set_xlim(0, 40)

# 6. 时频图 - 实验 By
f2, t2, Zxx2 = signal.stft(By, fs, nperseg=1024, noverlap=512)
mask = (f2 <= 40)
im2 = ax2.pcolormesh(t2, f2[mask], np.abs(Zxx2[mask]), shading='gouraud', cmap='bwr')
# ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Frequency [Hz]')
ax2.set_title('By Time-Frequency Analysis')
fig.colorbar(im2, ax=ax2, label='Magnitude')
im2.set_clim(0, 0.25)
ax2.set_ylim(1.5, 10)
ax2.set_xlim(0, 40)

# 7. 时频图 - 空载 Bx
f3, t3, Zxx3 = signal.stft(empty_Bx, fs, nperseg=1024, noverlap=512)
mask = (f3 <= 40)
im3 = ax3.pcolormesh(t3, f3[mask], np.abs(Zxx3[mask]), shading='gouraud', cmap='bwr')
ax3.set_ylabel('Frequency [Hz]')
ax3.set_title('Empty Bx Time-Frequency Analysis')
fig.colorbar(im3, ax=ax3, label='Magnitude')
im3.set_clim(0, 0.25)
ax3.set_ylim(1.5, 10)
ax3.set_xlim(0, 40)

# 8. 时频图 - 空载 By
f4, t4, Zxx4 = signal.stft(empty_By, fs, nperseg=1024, noverlap=512)
mask = (f4 <= 40)
im4 = ax4.pcolormesh(t4, f4[mask], np.abs(Zxx4[mask]), shading='gouraud', cmap='bwr')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Frequency [Hz]')
ax4.set_title('Empty By Time-Frequency Analysis')
fig.colorbar(im4, ax=ax4, label='Magnitude')
im4.set_clim(0, 0.25)
ax4.set_ylim(1.5, 10)
ax4.set_xlim(0, 40)

plt.tight_layout()
plt.show()
