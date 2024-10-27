#!/bin/bash

# 创建第一个 tmux 会话，并运行 visdom
tmux new-session -d -s visdom 'visdom'

# 创建第二个 tmux 会话，运行 tmux，然后执行 Python 脚本
tmux new-session -d -s p1 'env CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0 xvfb-run -a -s "-screen 0 1400x900x24" python experiments/gym/run_se_control.py'

# 创建第三个 tmux 会话，运行 tmux，然后执行 Python 脚本
tmux new-session -d -s p2 'env CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=2 xvfb-run -a -s "-screen 0 1400x900x24" python experiments/gym/run_se_control.py'

# 创建第四个 tmux 会话，执行 nvitop
#tmux new-session -d -s nvitop 'nvitop'

# 创建第五个 tmux 会话，执行 htop
#tmux new-session -d -s htop 'htop'
tmux attach -t p1
#tmux attach -t p2


echo "所有会话已启动。使用 'tmux attach -t <session_name>' 连接到指定会话。"
