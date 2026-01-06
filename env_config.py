"""
环境配置文件 - 方便切换不同环境
"""

# 环境配置字典
ENV_CONFIGS = {
    'acrobot': {
        'env_name': 'acrobot',
        'obs_dim': 6,  # cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot
        'action_dim': 3,  # -1, 0, +1 扭矩
        'action_meanings': ['Torque -1', 'No Torque', 'Torque +1'],
        'description': 'Acrobot双摆控制任务，目标是将自由端摆到目标高度'
    },
    
    'mountaincar': {
        'env_name': 'mountaincar', 
        'obs_dim': 2,  # position, velocity
        'action_dim': 3,  # 左加速, 不加速, 右加速
        'action_meanings': ['Accelerate Left', 'No Acceleration', 'Accelerate Right'],
        'description': 'MountainCar爬山车任务，目标是通过左右加速到达山顶'
    },
    
    'pendulum': {
        'env_name': 'pendulum',
        'obs_dim': 3,  # cos(θ), sin(θ), θ_dot  
        'action_dim': 1,  # 连续扭矩 (但会被离散化为4个动作)
        'action_meanings': ['Torque -2', 'Torque -0.67', 'Torque +0.67', 'Torque +2'],
        'description': 'Pendulum倒立摆任务，目标是保持摆杆直立'
    },
    
    'cartpole': {
        'env_name': 'cartpole',
        'obs_dim': 4,  # x, x_dot, theta, theta_dot
        'action_dim': 2,  # 向左, 向右
        'action_meanings': ['Push Left', 'Push Right'],
        'description': 'CartPole平衡控制任务，目标是保持杆的平衡'
    }
}

def get_env_config(env_name):
    """获取环境配置"""
    if env_name in ENV_CONFIGS:
        return ENV_CONFIGS[env_name]
    else:
        raise ValueError(f"未知环境: {env_name}. 支持的环境: {list(ENV_CONFIGS.keys())}")

def print_env_info(env_name):
    """打印环境信息"""
    config = get_env_config(env_name)
    print(f"\n=== {env_name.upper()} 环境信息 ===")
    print(f"描述: {config['description']}")
    print(f"观测维度: {config['obs_dim']}")
    print(f"动作维度: {config['action_dim']}")
    print(f"动作含义: {config['action_meanings']}")
    print("=" * 40)

def print_all_envs():
    """打印所有支持的环境"""
    print("\n=== 支持的环境列表 ===")
    for env_name in ENV_CONFIGS.keys():
        print_env_info(env_name)

if __name__ == "__main__":
    print_all_envs()
