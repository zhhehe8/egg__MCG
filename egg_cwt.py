import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

##   该脚本用于1个txt文件

# 1. 加载数据
data = np.loadtxt(r'C:\Users\Administrator\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t3.txt', skiprows=2, encoding="utf-8")  # 跳过头部描述行
Bx = data[:, 0]  # 第一列数据
By = data[:, 1]  # 第二列数据
fs = 1000  # 采样率

# 加载空载数据"20250218_空载1.txt"
empty_data = np.loadtxt(r'C:\Users\Administrator\Desktop\egg_2025\鸡蛋空载\空载1.txt', skiprows=2, encoding="utf-8")
empty_Bx = empty_data[:, 0]
empty_By = empty_data[:, 1]


# 2. 创建画布
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
ax1, ax2 = axs[0, 0], axs[0, 1]  # 第一列
ax3, ax4 = axs[1, 0], axs[1, 1]  # 第二列

# 3. 处理第一列数据Bx
f1, t1, Zxx1 = signal.stft(Bx, fs, nperseg=1024, noverlap=512)
mask = (f1 <= 40)  # 筛选0-40Hz
im1 = ax1.pcolormesh(t1, f1[mask], np.abs(Zxx1[mask]), 
                    shading='gouraud', cmap='bwr')
ax1.set_ylabel('Frequency [Hz]')
ax1.set_title('Bx Time-Frequency Analysis')
fig.colorbar(im1, ax=ax1, label='Magnitude')
# 设置colorbar范围为0-1Hz
im1.set_clim(0, 0.25)

# 设置y轴范围为1-6Hz
ax1.set_ylim(1.5, 10)
# 设置x轴范围为0-40s
ax1.set_xlim(0, 40)

# 设置空载数据Bx
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
ax3.set_ylim(1.5, 10)

ax3.set_xlim(0, 40)


# 4. 处理第二列数据
f2, t2, Zxx2 = signal.stft(By, fs, nperseg=1024, noverlap=512)
mask = (f2 <= 40)  # 筛选0-40Hz
im2 = ax2.pcolormesh(t2, f2[mask], np.abs(Zxx2[mask]),
                    shading='gouraud', cmap='bwr')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Frequency [Hz]')
ax2.set_title('By Time-Frequency Analysis')
fig.colorbar(im2, ax=ax2, label='Magnitude')
# 设置colorbar范围为0-1Hz
im2.set_clim(0, 0.25)

# 设置y轴范围为1-6Hz
ax2.set_ylim(1.5, 10)
# 设置x轴范围为0-40s
ax2.set_xlim(0, 40)

# 设置空载数据By
f4, t4, Zxx4 = signal.stft(empty_By, fs, nperseg=1024, noverlap=512)
mask = (f4 <= 40)
im4 = ax4.pcolormesh(t4, f4[mask], np.abs(Zxx4[mask]), 
                      shading='gouraud', cmap='bwr')
ax4.set_ylabel('Frequency [Hz]')
ax4.set_title('Empty By Time-Frequency Analysis')
fig.colorbar(im4, ax=ax4, label='Magnitude')

im4.set_clim(0, 0.25)

ax4.set_ylim(1.5, 10)
ax4.set_xlim(0, 40)


plt.show()