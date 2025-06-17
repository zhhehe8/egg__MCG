# mcg_processing.py
"""包含所有数据处理算法"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks, stft
from fastdtw import fastdtw
import pywt
from pathlib import Path
from typing import Tuple, List, Optional

# --- 数据加载 ---
def load_cardiac_data(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """从文本文件加载心脏数据 (Bx, By)。"""
    try:
        data = np.loadtxt(filepath, skiprows=2, encoding="utf-8")
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"错误: 数据文件 {filepath.name} 需要至少两列。")
            return None, None
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"错误: 加载数据 '{filepath}' 时出错: {e}")
        return None, None

# --- 滤波函数 ---
def apply_bandpass_filter(data: np.ndarray, fs: int, order: int, lowcut: float, highcut: float) -> np.ndarray:
    """应用巴特沃斯带通滤波器。"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def apply_notch_filter(data: np.ndarray, fs: int, freq: float, q_factor: float) -> np.ndarray:
    """应用陷波滤波器去除工频干扰。"""
    nyquist = 0.5 * fs
    b, a = iirnotch(freq / nyquist, q_factor)
    return filtfilt(b, a, data)

def apply_wavelet_denoise(data: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """应用小波变换进行去噪。"""
    coeffs = pywt.wavedec(data, wavelet, mode='per', level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    reconstructed = pywt.waverec(new_coeffs, wavelet, mode='per')
    return reconstructed[:len(data)] # 确保长度一致

# --- R峰检测与定位 ---
def find_r_peaks(data: np.ndarray, fs: int, min_height_factor: float, min_distance_ms: int) -> np.ndarray:
    """在信号中寻找R峰的整数索引。"""
    if data is None or len(data) == 0:
        return np.array([])
    min_height = min_height_factor * np.max(data)
    min_distance = int(min_distance_ms / 1000 * fs)
    peaks, _ = find_peaks(data, height=min_height, distance=min_distance)
    return peaks

def interpolate_r_peaks(data: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    """通过抛物线插值，将R峰定位提升到亚样本精度。"""
    precise_indices = []
    for peak_idx in peak_indices:
        if 1 <= peak_idx < len(data) - 1:
            y = data[peak_idx-1 : peak_idx+2]
            x = np.arange(-1, 2)
            coeffs = np.polyfit(x, y, 2)
            if coeffs[0] == 0: # 避免除以零
                precise_indices.append(float(peak_idx))
                continue
            vertex_x = -coeffs[1] / (2 * coeffs[0])
            precise_indices.append(peak_idx + vertex_x)
        else:
            precise_indices.append(float(peak_idx))
    return np.array(precise_indices)

# --- 叠加平均算法 ---
def _extract_beats(signal_data: np.ndarray, precise_peak_indices: np.ndarray, pre_samples: int, post_samples: int) -> List[np.ndarray]:
    """(内部函数)基于精确R峰索引提取所有心拍。"""
    beats_list = []
    original_x = np.arange(len(signal_data))
    relative_x = np.arange(-pre_samples, post_samples)
    for peak_loc in precise_peak_indices:
        absolute_x = peak_loc + relative_x
        if absolute_x[0] < 0 or absolute_x[-1] >= len(signal_data):
            continue
        interpolated_beat = np.interp(absolute_x, original_x, signal_data)
        beats_list.append(interpolated_beat)
    return beats_list

def get_median_beat(all_beats: List[np.ndarray]) -> Optional[np.ndarray]:
    """计算中位数平均心拍。"""
    if not all_beats:
        return None
    return np.median(np.array(all_beats), axis=0)

def get_dtw_beat(all_beats: List[np.ndarray]) -> Optional[np.ndarray]:
    """计算DTW对齐后的平均心拍。"""
    if len(all_beats) < 2:
        return np.array(all_beats[0]) if all_beats else None

    beats_array = np.array(all_beats)
    template_beat = np.median(beats_array, axis=0)
    
    warped_beats = []
    print(f"开始对 {len(beats_array)} 个心拍进行DTW对齐...")
    for i, beat in enumerate(beats_array):
        _, path = fastdtw(template_beat, beat, dist=lambda a, b: (a - b)**2)
        
        warped_beat = np.zeros_like(template_beat)
        warp_counts = np.zeros_like(template_beat, dtype=int)
        
        for template_idx, beat_idx in path:
            warped_beat[template_idx] += beat[beat_idx]
            warp_counts[template_idx] += 1
            
        warp_counts[warp_counts == 0] = 1
        warped_beat /= warp_counts
        warped_beats.append(warped_beat)
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(beats_array)}...")
            
    return np.mean(np.array(warped_beats), axis=0)


""" 计算时频图（stft 和 cwt） """
def calculate_stft(data: np.ndarray, fs: int, window_length: int, overlap_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算短时傅里叶变换 (STFT)。"""
    noverlap = int(window_length * overlap_ratio)
    f, t, Zxx = stft(data, fs, nperseg=window_length, noverlap=noverlap)
    return f, t, np.abs(Zxx)

def calculate_cwt(data: np.ndarray, fs: int, wavelet: str, max_scale: int, num_scales: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算连续小波变换 (CWT)。"""
    # 创建一个尺度数组
    scales = np.arange(1, max_scale)
    # 计算CWT
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=1/fs)
    return frequencies, np.abs(coefficients)