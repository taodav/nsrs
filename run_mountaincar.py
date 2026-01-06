"""
运行MountainCar环境的脚本
"""
import sys
import os

# 添加路径
sys.path.append('.')
sys.path.append('experiments/gym')

# 导入运行脚本
from experiments.gym.run_se_control import *

if __name__ == "__main__":
    # 修改默认参数为MountainCar
    Defaults.ENV = 'mountaincar'
    Defaults.HIGHER_DIM_OBS = True  # 支持高维图像观测
    Defaults.OBS_PER_STATE = 4      # 4个连续状态
    Defaults.EPOCHS = 1
    Defaults.STEPS_PER_EPOCH = 1000
    Defaults.MONITOR = True
    
    print("=== 启动MountainCar环境训练 ===")
    print(f"环境: {Defaults.ENV}")
    print(f"高维观测: {Defaults.HIGHER_DIM_OBS}")
    print(f"历史状态数: {Defaults.OBS_PER_STATE}")
    print(f"训练轮数: {Defaults.EPOCHS}")
    print(f"每轮步数: {Defaults.STEPS_PER_EPOCH}")
    print("================================")
