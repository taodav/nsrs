import json
import matplotlib.pyplot as plt
import os

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

def plot_coverage(x_data, y_data):
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label='Coverage')
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('Coverage Data Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_data_to_file(x_data, y_data, output_filename):
    # 获取当前脚本目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_filename)
    
    # 将数据保存到JSON文件中
    with open(output_path, 'w') as f:
        json.dump({'x_data': x_data, 'y_data': y_data}, f)
    print(f"数据已保存到 {output_path}")

def process_files_in_directory(root_dir, save_dir):
    # 检查目标保存目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 遍历根目录下的所有文件和文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if 'plot' in filename:  # 检查文件名中是否包含 'plot'
                file_path = os.path.join(dirpath, filename)
                

                # 提取数据
                x_data, y_data = extract_coverage_data(file_path)
                if len(x_data) < 10:
                    continue
                print(f"找到文件: {file_path}")
                # 定义输出文件路径
                output_filename = os.path.splitext(filename)[0] + '_extracted.json'
                output_path = os.path.join(save_dir, output_filename)
                
                # 保存数据
                save_data_to_file(x_data, y_data, output_path)
                plot_coverage(x_data, y_data)

def plot_single_file(filename):
    x_data, y_data = extract_coverage_data(filename)
    
    if x_data and y_data:
        output_filename = 'coverage.json'
        save_data_to_file(x_data, y_data, output_filename)
        plot_coverage(x_data, y_data)
    else:
        print("没有找到标题为'coverage'的数据。")
    

if __name__ == "__main__":
    # filename = '/home/user/Downloads/renyi2_sigma05_alpha101_avg370_std152_unstable1400/renyi2_sigma05_alpha101_312'  # 替换为你的文件名
    #处理单个文件
    # filename = "C:/Users/15210/Desktop/nsrs测试数据/renyi4/renyi4_sigma05_alpha101_489/acrobot--novelty_reward_with_d_step_q_planning_2024-09-02--08-07-31_0/plot"
    # plot_single_file(filename)
    

    #处理这个目录下的文件
    root_directory = 'C:/Users/15210/Desktop/nsrs测试数据'  # 替换为你的根目录路径
    save_directory = '/home/user/Desktop/nsrs/examples/gym/experiments/acrobot--novelty_reward_with_d_step_q_planning_2024-08-29--11-08-02_0/plot'
    
    process_files_in_directory(root_directory, save_directory)



