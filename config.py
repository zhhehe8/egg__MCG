# config.py
"""包含所有可变参数（文件路径、滤波器设置、算法参数等等）"""

from pathlib import Path

# -----------------
# 1. 文件路径配置
# -----------------
# 使用 Path 对象，这比简单的字符串更健壮，能更好地处理跨平台路径问题
BASE_DIR = Path('/Users/yanchen/Desktop/Projects/egg_2025') # 项目的基础目录
INPUT_FILE = BASE_DIR / 'B_egg' / 'B_egg_d20' / 'egg_d20_B30_t1_待破壳.txt' # 输入数据文件
OUTPUT_DIR = BASE_DIR / 'Figures' # 所有输出（如图片）的保存目录

# -----------------
# 2. 信号预处理配置
# -----------------
PROCESSING_PARAMS = {
    'fs': 1000,  # 采样率 (Hz)
    'reverse_Bx': False, # 是否反转Bx信号
    'reverse_By': False, # 是否反转By信号
}

# -----------------
# 3. 滤波器配置
# -----------------
FILTER_PARAMS = {
    'bandpass': {
        'order': 4,
        'lowcut': 0.5,   # Hz, 低截止频率
        'highcut': 45.0, # Hz, 高截止频率
    },
    'notch': {
        'freq': 50.0,    # Hz, 工频干扰频率
        'q_factor': 30.0,
    },
    'wavelet': { # 小波去噪配置
        'enabled': False, # 设置为 True 来启用小波去噪
        'wavelet': 'sym8',
        'level': 6,
    }
}

# -----------------
# 4. R峰检测配置
# -----------------
R_PEAK_PARAMS = {
    'min_height_factor': 0.4, # R峰最小高度因子
    'min_distance_ms': 200,   # R峰最小距离 (毫秒)
}

# -----------------
# 5. 叠加平均配置
# -----------------
AVERAGING_PARAMS = {
    'pre_r_ms': 100,  # R峰前的时间窗口 (毫秒)
    'post_r_ms': 300, # R峰后的时间窗口 (毫秒)
}


# -----------------
# 6. 时频分析配置
# -----------------
TIME_FREQUENCY_PARAMS = {
    # 'enabled_method' 选项:
    # 'stft' - 只进行短时傅里叶变换
    # 'cwt'  - 只进行连续小波变换
    # 'both' - 两种方法都进行
    # 'none' - 不进行时频分析
    'enabled_method': 'stft',

    'stft': {
        'window_length': 256,       # STFT窗口长度
        'overlap_ratio': 0.5,       # 窗口重叠比例 (例如0.5代表50%)
    },
    'cwt': {
        'wavelet': 'cmor1.5-1.0',    # CWT使用的小波基函数
        'max_scale': 128,           # CWT分析的最大尺度 (值越大，频率越低)
        'num_scales': 100,          # 在1到max_scale之间取多少个尺度
    },
    'plot': {
        # 'style' 选项: 'single', 'dual', 'both', 'none'
        'style': 'dual'
    }
}


# -----------------
# 7. 绘图配置
# -----------------
PLOTTING_PARAMS = {
    # 'style' 选项:
    # 'single' - 每个通道单独生成一张图
    # 'dual'   - Bx和By整合到一张图中显示
    # 'both'   - 以上两种图都生成
    # 'none'   - 不生成任何图表
    'filtering_plot': {
        'style': 'dual', 
    },
    'averaging_plot': {
        'style': 'dual',
    }
}

