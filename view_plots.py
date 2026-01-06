import os
from nsrl.helper.plot import replay_plot
import subprocess
import time

def main():
    # 启动visdom服务器
    print("Starting visdom server...")
    subprocess.Popen(["python", "-m", "visdom.server"])
    
    # 等待服务器启动
    time.sleep(2)
    
    # 遍历examples文件夹下的所有实验记录
    examples_dir = "examples"
    for root, dirs, files in os.walk(examples_dir):
        for dir in dirs:
            if "experiments" in dir:
                exp_path = os.path.join(root, dir)
                for exp_name in os.listdir(exp_path):
                    plot_path = os.path.join(exp_path, exp_name, "plot")
                    if os.path.exists(plot_path):
                        print(f"Loading plot from: {plot_path}")
                        try:
                            replay_plot(plot_path)
                        except Exception as e:
                            print(f"Error loading plot {plot_path}: {str(e)}")

    print("\nAll plots have been loaded!")
    print("You can view them at http://localhost:8097")
    print("Press Ctrl+C to exit")
    
    # 保持脚本运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    # main()
    plot_path = "examples/gym/experiments/acrobot--novelty_reward_with_d_step_q_planning_2024-11-20--14-30-05_0/plot"
    # plot_path2 = "C:\\Users\\15210\\Desktop\\nsrs\\examples\\pycolab\\experiments\\N_maze novelty reward with d step q planning_2024-11-28 00-31-25_0"
    replay_plot(plot_path)
