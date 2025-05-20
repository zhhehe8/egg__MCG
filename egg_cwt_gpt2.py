import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# 1. 加载数据 (Load Data)
# --- Make sure these paths are correct on your system ---
# experiment_file_path = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t3.txt'
# empty_load_file_path = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\鸡蛋空载\空载1.txt'
# --- ---

experiment_file_path = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\朱鹮_250426\朱鹮day25_2_t3.txt'
empty_load_file_path = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\朱鹮_250426\空载1.txt'

# --- Error handling for file loading ---
if not os.path.exists(experiment_file_path):
    raise FileNotFoundError(f"Experiment data file not found: {experiment_file_path}")
if not os.path.exists(empty_load_file_path):
    raise FileNotFoundError(f"Empty load data file not found: {empty_load_file_path}")
# --- ---

data = np.loadtxt(experiment_file_path, skiprows=2, encoding="utf-8")
Bx = data[:, 0]
By = data[:, 1]
fs = 1000  # 采样率 (Sampling Rate)

# 加载空载数据 (Load Empty Load Data)
empty_data = np.loadtxt(empty_load_file_path, skiprows=2, encoding="utf-8")
empty_Bx = empty_data[:, 0]
empty_By = empty_data[:, 1]

# --- Ensure data lengths are reasonable ---
# If empty load data is much longer/shorter, consider truncating/padding
# For now, we assume they are comparable or analysis on full length is desired.
# Example truncation (uncomment if needed):
# min_len = min(len(Bx), len(empty_Bx))
# Bx = Bx[:min_len]
# By = By[:min_len]
# empty_Bx = empty_Bx[:min_len]
# empty_By = empty_By[:min_len]
# --- ---

# 2. 数据融合 (Data Fusion) - Calculate vector magnitude
B_combined = np.sqrt(Bx**2 + By**2)
empty_B_combined = np.sqrt(empty_Bx**2 + empty_By**2)

# 3. 创建画布：1行2列 (Create Canvas: 1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True) # sharey might be useful
ax_exp = axs[0]       # Left plot: Combined Experiment STFT
ax_empty = axs[1]     # Right plot: Combined Empty Load STFT

# STFT parameters (consistent)
nperseg_val = 1024
noverlap_val = 512
freq_limit = 40
ylim_freq = (1.5, 10)
clim_val = (0, 0.25) # Keep original clim, adjust if combined magnitude is very different
cmap_val = 'bwr'     # Or 'viridis', 'magma', etc.

# 4. 时频图 - 融合后的实验数据 (Time-Frequency Plot - Combined Experiment Data)
f_exp, t_exp, Zxx_exp = signal.stft(B_combined, fs, nperseg=nperseg_val, noverlap=noverlap_val)
mask_exp = (f_exp <= freq_limit)
im_exp = ax_exp.pcolormesh(t_exp, f_exp[mask_exp], np.abs(Zxx_exp[mask_exp]),
                           shading='gouraud', cmap=cmap_val, vmin=clim_val[0], vmax=clim_val[1])
ax_exp.set_ylabel('Frequency [Hz]')
ax_exp.set_xlabel('Time [s]')
ax_exp.set_title(r'Time-frequency results of day20') # Use LaTeX for formula
fig.colorbar(im_exp, ax=ax_exp, label='Magnitude')
#im_exp.set_clim(clim_val) # Alternative way to set clim
ax_exp.set_ylim(ylim_freq)
ax_exp.set_xlim(0, 40) 

# 5. 时频图 - 融合后的空载数据 (Time-Frequency Plot - Combined Empty Load Data)
f_empty, t_empty, Zxx_empty = signal.stft(empty_B_combined, fs, nperseg=nperseg_val, noverlap=noverlap_val)
# Check if empty load signal is long enough for STFT with these settings
if Zxx_empty.size == 0:
     print(f"Warning: Empty load signal might be too short for STFT with nperseg={nperseg_val}")
     ax_empty.set_title(r'Time-frequency results of no load')
     ax_empty.set_xlabel('Time [s]')

else:
    mask_empty = (f_empty <= freq_limit)
    im_empty = ax_empty.pcolormesh(t_empty, f_empty[mask_empty], np.abs(Zxx_empty[mask_empty]),
                                   shading='gouraud', cmap=cmap_val, vmin=clim_val[0], vmax=clim_val[1])
    ax_empty.set_xlabel('Time [s]')
    # ax_empty.set_ylabel('Frequency [Hz]') # Removed as sharey=True
    ax_empty.set_title(r'Time-frequency results of no load') # Use LaTeX for formula
    fig.colorbar(im_empty, ax=ax_empty, label='Magnitude')
    #im_empty.set_clim(clim_val) # Alternative way to set clim
    ax_empty.set_ylim(ylim_freq)
    ax_empty.set_xlim(0, 60)

plt.tight_layout()
plt.show()
fig.savefig(r'C:\Users\Xiaoning Tan\Desktop\egg_figure\time_frequency_results_day20.png', dpi=300)

print("Figure saved successfully.")