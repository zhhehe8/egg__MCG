import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


"""" 展示一组数据x,y轴的时频图，包含实验数据和空载数据的对比。""" 

# 1. 加载数据
# Make sure to update the file paths if they are different on your system
try:
    # Ensure these paths are correct and accessible in your environment
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

except FileNotFoundError as e:
    print(f"Error: File not found. Please check the file paths. Details: {e}")
    # Depending on how you want to handle this, you might exit or raise the error
    # For this example, we'll print and exit if files are critical for the script's purpose
    # If running in an environment where you expect the user to provide files,
    # this kind of error handling is crucial.
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()


# 2. 创建画布：2行2列
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
# Assign axes for the 2x2 grid
ax1, ax2 = axs[0]      # 第一行：实验数据时频图 (Bx, By)
ax3, ax4 = axs[1]      # 第二行：空载数据时频图 (Bx, By)

# 3. 时频图 - 实验 Bx
f1, t1, Zxx1 = signal.stft(Bx, fs, nperseg=1024, noverlap=512)
mask1 = (f1 <= 40)
# Ensure t1 and f1[mask1] are not empty before plotting
if t1.size > 0 and f1[mask1].size > 0 and Zxx1[mask1].size > 0:
    im1 = ax1.pcolormesh(t1, f1[mask1], np.abs(Zxx1[mask1]), shading='gouraud', cmap='bwr')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_title('Experiment Bx Time-Frequency Analysis')
    fig.colorbar(im1, ax=ax1, label='Magnitude')
    im1.set_clim(0, 0.25)
    ax1.set_ylim(1.5, 10)
    ax1.set_xlim(0, t1.max() if t1.max() < 40 else 40) # Adjust xlim based on data
else:
    ax1.set_title('Experiment Bx Time-Frequency Analysis (No data to plot)')

# 4. 时频图 - 实验 By
f2, t2, Zxx2 = signal.stft(By, fs, nperseg=1024, noverlap=512)
mask2 = (f2 <= 40)
if t2.size > 0 and f2[mask2].size > 0 and Zxx2[mask2].size > 0:
    im2 = ax2.pcolormesh(t2, f2[mask2], np.abs(Zxx2[mask2]), shading='gouraud', cmap='bwr')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_title('Experiment By Time-Frequency Analysis')
    fig.colorbar(im2, ax=ax2, label='Magnitude')
    im2.set_clim(0, 0.25)
    ax2.set_ylim(1.5, 10)
    ax2.set_xlim(0, t2.max() if t2.max() < 40 else 40)
else:
    ax2.set_title('Experiment By Time-Frequency Analysis (No data to plot)')

# 5. 时频图 - 空载 Bx
f3, t3, Zxx3 = signal.stft(empty_Bx, fs, nperseg=1024, noverlap=512)
mask3 = (f3 <= 40)
if t3.size > 0 and f3[mask3].size > 0 and Zxx3[mask3].size > 0:
    im3 = ax3.pcolormesh(t3, f3[mask3], np.abs(Zxx3[mask3]), shading='gouraud', cmap='bwr')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Frequency [Hz]')
    ax3.set_title('Empty Bx Time-Frequency Analysis')
    fig.colorbar(im3, ax=ax3, label='Magnitude')
    im3.set_clim(0, 0.25)
    ax3.set_ylim(1.5, 10)
    ax3.set_xlim(0, t3.max() if t3.max() < 40 else 40)
else:
    ax3.set_title('Empty Bx Time-Frequency Analysis (No data to plot)')


# 6. 时频图 - 空载 By
f4, t4, Zxx4 = signal.stft(empty_By, fs, nperseg=1024, noverlap=512)
mask4 = (f4 <= 40)
if t4.size > 0 and f4[mask4].size > 0 and Zxx4[mask4].size > 0:
    im4 = ax4.pcolormesh(t4, f4[mask4], np.abs(Zxx4[mask4]), shading='gouraud', cmap='bwr')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Frequency [Hz]')
    ax4.set_title('Empty By Time-Frequency Analysis')
    fig.colorbar(im4, ax=ax4, label='Magnitude')
    im4.set_clim(0, 0.25)
    ax4.set_ylim(1.5, 10)
    ax4.set_xlim(0, t4.max() if t4.max() < 40 else 40)
else:
    ax4.set_title('Empty By Time-Frequency Analysis (No data to plot)')


plt.tight_layout()
plt.show()
