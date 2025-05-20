import os
# from egg_module import batch_process
from egg_module import batch_process
if __name__ == "__main__":
    # 配置路径
    input_folder = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d11'  # 原始数据目录
    output_folder = r'C:\Users\Xiaoning Tan\Desktop\Begg_cwt_output\Begg_cwt_d11'  # 输出图片目录
   
    # 执行批量处理
    batch_process(input_folder, output_folder)
    print("批量处理完成！")
