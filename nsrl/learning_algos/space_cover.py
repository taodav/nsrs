import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import binned_statistic_dd
import umap
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px

def compute_space_coverage(data, dx):
    """
    归一化数据、离散化并计算空间覆盖率。
    
    参数:
    data: ndarray, 输入数据，形状为(N, dim)
    dx: float, 离散化步长
    
    返回:
    coverage_rate: float, 空间覆盖率
    """
    # 步骤1: 归一化数据到 [0, 1]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    # 步骤2: 计算每个维度的bins数量
    n_dims = normalized_data.shape[1]
    bins = [int(1 / dx) for _ in range(n_dims)]  # 因为数据已经归一化，范围为 [0, 1]
    
    # 步骤3: 离散化并统计非空网格数
    hist, edges, binnumber = binned_statistic_dd(normalized_data, None, statistic='count', bins=bins, range=[[0, 1]]*n_dims)
    
    # 计算总的网格数量
    total_grids = np.prod(bins)
    
    # 计算非空的网格数
    non_empty_bins = np.count_nonzero(hist)
    
    # 计算空间覆盖率
    coverage_rate = non_empty_bins / total_grids
    
    return coverage_rate

def umap_visualization(data):
    '''data: [N, dim]'''
    data = np.random.rand(num, dim)  # 替换为实际数据

    # 创建UMAP对象
    reducer = umap.UMAP(n_neighbors=15)

    # 执行降维
    embedding = reducer.fit_transform(data)

    # 可视化
    #使用Plotly进行可视化
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        title='UMAP Dimensionality Reduction'
    )

    fig.update_layout(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2'
    )

    # 显示图形
    fig.show()


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    # 生成示例数据，替换为你的数据
    num = 1000
    dim = 4
    data = np.random.rand(num, dim)  # 替换为实际数据

    # 创建UMAP对象
    reducer = umap.UMAP(n_neighbors=15)

    # 执行降维
    embedding = reducer.fit_transform(data)

    # 可视化
    #使用Plotly进行可视化
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        title='UMAP Dimensionality Reduction'
    )

    fig.update_layout(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2'
    )

    # 显示图形
    fig.show()
