import json
import matplotlib.pyplot as plt
import os
import numpy as np

def extract_coverage_data(filename):
    x_data, y_data = [], []

    # 读取文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 逐行解析数据
    for line in lines:
        try:
            # 转换为Python字典
            data = json.loads(line)
            opts = data[1].get('opts', {})
            title = opts.get('title', '').lower()

            # 判断标题是否为'coverage'
            if 'coverage' in title:
                # 提取x和y数据
                x = data[1]['data'][0]['x']
                y = data[1]['data'][0]['y']
                x_data.extend(x)
                y_data.extend(y)

        except (json.JSONDecodeError, IndexError, KeyError):
            # 跳过无法解析的行或缺少字段的行
            continue

    return x_data, y_data

def process_files_in_directory(root_dir):
    all_x_data, all_y_data = [], []

    # 遍历根目录下的所有文件和文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if 'plot' in filename:  # 检查文件名中是否包含 'plot'
                file_path = os.path.join(dirpath, filename)
                print(f"找到文件: {file_path}")

                # 提取数据
                x_data, y_data = extract_coverage_data(file_path)
                if x_data and y_data:
                    all_x_data.extend(x_data)
                    all_y_data.extend(y_data)

    return all_x_data, all_y_data

def plot_coverage(x_data, y_data):
    if not x_data or not y_data:
        print("没有数据可用于绘图。")
        return

    # 将数据转换为NumPy数组以计算均值和标准差
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # 计算均值和标准差
    mean_y = np.mean(y_data)
    std_y = np.std(y_data)

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label='Coverage')
    plt.axhline(mean_y, color='r', linestyle='-', label=f'Mean: {mean_y:.2f}')
    plt.fill_between(x_data, mean_y - std_y, mean_y + std_y, color='r', alpha=0.2, label=f'Standard Deviation: {std_y:.2f}')
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('Coverage Data Plot with Mean and Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    root_directory = '/home/user/Desktop/nsrs/data'  # 替换为你的根目录路径

    # 处理目录中的文件
    x_data, y_data = process_files_in_directory(root_directory)

    # print("x_data shape:", x_data.shape)
    breakpoint()
    # 绘制图表
    plot_coverage(x_data, y_data)
