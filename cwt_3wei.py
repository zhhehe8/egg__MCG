"""
绘制3维的时频图 (频率为X轴, 时间为Y轴)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import os

# --- Configuration Parameters ---
# Data loading
# experimental_file_path = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\朱鹮_250426\朱鹮day25_2_t2.txt'
experimental_file_path = '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t1_待破壳.txt'
fs = 1000  # 采样率

# STFT parameters
nperseg_val = 512
noverlap_val = nperseg_val // 2

# Frequency range for analysis and plotting (New X-axis)
plot_freq_min = 1  # Hz
plot_freq_max = 6.0 # Hz

# Time range for display (New Y-axis)
time_limit_display = 100 # seconds

# Magnitude (Z-axis) and Colormap limits
magnitude_min_clim = 0.0
magnitude_max_clim = 10

# 3D Viewpoint
elevation_angle = 25
azimuthal_angle = -55 # You might want to adjust this after swapping axes

# Surface plot parameters
surface_cmap = 'viridis'  # Colormap for the surface plot
# Strides for the *surface plot* after Z matrix (Magnitude) is prepared.
# If Z is (num_times, num_frequencies):
#   rstride_surf will sample along time.
#   cstride_surf will sample along frequency.
# User's original intent: row_stride=1 for frequency detail, col_stride=5 for time detail.
# So, for Z.T (Mag_for_plot.T):
#   Stride for time dimension (rows of Z.T) should be based on user's old 'col_stride'
#   Stride for frequency dimension (cols of Z.T) should be based on user's old 'row_stride'
config_time_stride = 5  # Detail along the new Y-axis (Time)
config_freq_stride = 1  # Detail along the new X-axis (Frequency)
# --- End Configuration ---

# 1. Load Data
if not os.path.exists(experimental_file_path):
    print(f"错误: 文件未找到 {experimental_file_path}")
    exit()
try:
    data = np.loadtxt(experimental_file_path, skiprows=2, encoding="utf-8")
    Bx = data[:, 0]
    By = data[:, 1]
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()

# 2. Create Figure and 3D Axes
fig = plt.figure(figsize=(18, 8))
ax1_3d = fig.add_subplot(1, 2, 1, projection='3d')
ax2_3d = fig.add_subplot(1, 2, 2, projection='3d')
fig.suptitle('3D Time-Frequency Analysis (Frequency vs. Time)', fontsize=16)

# --- Helper function for plotting ---
def plot_3d_spectrogram_freq_vs_time(ax, S_data, fs_data, title_text):
    f_stft, t_stft, Zxx = signal.stft(S_data, fs_data, nperseg=nperseg_val, noverlap=noverlap_val)
    Mag_stft = np.abs(Zxx) # Shape: (num_frequencies_stft, num_times_stft)

    # Filter data for the surface plot to the desired frequency range (new X-axis)
    freq_plot_mask = (f_stft >= plot_freq_min) & (f_stft <= plot_freq_max)
    f_for_plot_axis = f_stft[freq_plot_mask]            # This will be our X-axis values
    Mag_freq_filtered = Mag_stft[freq_plot_mask, :] # Shape: (num_selected_freq, num_times_stft)

    # Time vector (t_stft) will be our Y-axis values
    t_for_plot_axis = t_stft

    if f_for_plot_axis.size == 0 or t_for_plot_axis.size == 0 or Mag_freq_filtered.size == 0:
        ax.set_title(f'{title_text}\n(No data in selected range)')
        return

    # Create meshgrid: X is Frequency, Y is Time
    F_mesh, T_mesh = np.meshgrid(f_for_plot_axis, t_for_plot_axis)
    # F_mesh shape: (len(t_for_plot_axis), len(f_for_plot_axis))
    # T_mesh shape: (len(t_for_plot_axis), len(f_for_plot_axis))

    # Z data (Magnitude) needs to be transposed to match meshgrid
    # Mag_freq_filtered is (num_selected_freq, num_times_stft)
    # We need Z of shape (len(t_for_plot_axis), len(f_for_plot_axis))
    Mag_for_surface = Mag_freq_filtered.T

    # Dynamic stride for the time dimension (new Y-axis, rows of Mag_for_surface)
    # Aim for roughly 150-250 polygons in the time dimension for the surface
    dynamic_time_stride = max(1, int(t_for_plot_axis.shape[0] / 200)) if t_for_plot_axis.shape[0] > 200 else config_time_stride
    # Stride for the frequency dimension (new X-axis, columns of Mag_for_surface)
    static_freq_stride = config_freq_stride


    surf = ax.plot_surface(F_mesh, T_mesh, Mag_for_surface, # X=Freq, Y=Time, Z=Mag
                           cmap=surface_cmap,
                           vmin=magnitude_min_clim,
                           vmax=magnitude_max_clim,
                           edgecolor='none',
                           rstride=dynamic_time_stride, # Stride along rows of Mag_for_surface (Time)
                           cstride=static_freq_stride,  # Stride along columns of Mag_for_surface (Frequency)
                           antialiased=True,
                           shade=True)

    ax.set_xlabel('Frequency (Hz)', labelpad=15) # New X-axis
    ax.set_ylabel('Time (s)', labelpad=15)       # New Y-axis
    ax.set_zlabel('Magnitude', labelpad=15)      # Z-axis remains Magnitude
    ax.set_title(title_text, pad=25)

    # Set axis limits
    current_t_max = t_for_plot_axis.max()
    ax.set_xlim(plot_freq_min, plot_freq_max)          # X-axis is Frequency
    ax.set_ylim(0, min(current_t_max, time_limit_display)) # Y-axis is Time
    ax.set_zlim(magnitude_min_clim, magnitude_max_clim * 1.05 if magnitude_max_clim > 0 else 0.1)

    # Beautify Panes and Grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.8))
    ax.yaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.8))
    ax.zaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.8))
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4)

    ax.view_init(elev=elevation_angle, azim=azimuthal_angle)

    # Add a colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.15, orientation='vertical')
    cbar.set_label('Magnitude', rotation=270, labelpad=20)
    if magnitude_max_clim > 0:
        num_cbar_ticks = 6
        cbar_ticks = np.linspace(magnitude_min_clim, magnitude_max_clim, num=num_cbar_ticks)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])
    else:
        cbar.set_ticks([magnitude_min_clim, magnitude_max_clim])

# 3. Plot for Bx
plot_3d_spectrogram_freq_vs_time(ax1_3d, Bx, fs, 'Experiment Bx (Freq vs. Time)')

# 4. Plot for By
plot_3d_spectrogram_freq_vs_time(ax2_3d, By, fs, 'Experiment By (Freq vs. Time)')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.show()

# --- Notes on Visualization Choices ---
# 1. Viewpoint (elevation_angle, azimuthal_angle):
#    After swapping axes, you might find that a different viewpoint
#    (e.g., elevation_angle = 30, azimuthal_angle = -120 or elevation_angle = 60, azimuthal_angle = -70)
#    provides a better perspective of the surface. Experiment with these values.

# 2. Strides (config_time_stride, config_freq_stride):
#    - config_time_stride controls the sampling density along the Time axis (new Y).
#    - config_freq_stride controls the sampling density along the Frequency axis (new X).
#    Smaller values mean more detail but slower rendering.