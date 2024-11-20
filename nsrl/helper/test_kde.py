import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
import pandas as pd
from exploration import *

def test_kde_novelty():
    """
    测试KDE对新颖状态和已知状态的评分情况
    """
    # 生成测试数据
    np.random.seed(42)
    n_dims = 4
    
    # 1. 生成memory buffer数据（模拟已经访问过的状态）
    n_memory = 1000
    memory_data = np.random.normal(0, 1, (n_memory, n_dims))
    
    # 2. 生成测试状态
    n_test = 100
    # 已知状态：直接从memory中采样
    known_states = memory_data[np.random.choice(n_memory, n_test//2, replace=False)]
    # 新颖状态：从不同分布生成
    novel_states = np.random.normal(3, 1, (n_test//2, n_dims))  # 明显不同的分布
    
    # 合并测试状态
    test_states = np.vstack([known_states, novel_states])
    
    # 转换为torch tensor
    memory_tensor = torch.FloatTensor(memory_data)
    states_tensor = torch.FloatTensor(test_states)
    
    # 计算rewards
    rewards = calculate_scores_kde(states_tensor, memory_tensor)
    
    # 分析结果
    known_rewards = rewards[:n_test//2]
    novel_rewards = rewards[n_test//2:]
    
    # 打印统计信息
    print("\nRewards Statistics:")
    print(f"Known States - Mean: {np.mean(known_rewards):.3f}, Std: {np.std(known_rewards):.3f}")
    print(f"Novel States - Mean: {np.mean(novel_rewards):.3f}, Std: {np.std(novel_rewards):.3f}")
    
    # 创建用于boxplot的DataFrame
    df_rewards = pd.DataFrame({
        'Reward Value': np.concatenate([known_rewards, novel_rewards]),
        'State Type': ['Known States'] * len(known_rewards) + ['Novel States'] * len(novel_rewards)
    })
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_rewards, x='State Type', y='Reward Value')
    plt.title('Rewards Distribution')
    
    # Density plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=known_rewards, label='Known States')
    sns.kdeplot(data=novel_rewards, label='Novel States')
    plt.title('Rewards Density')
    plt.xlabel('Reward Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return np.mean(known_rewards), np.mean(novel_rewards)

if __name__ == '__main__':
    known_mean, novel_mean = test_kde_novelty()