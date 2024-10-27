import os
import re
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 设置你的视频文件所在的目录
directory = "C:/Users/15210/Desktop/nsrs测试数据/renyi0+kde/renyi0_kde_1214/acrobot--novelty_reward_with_d_step_q_planning_2024-09-04--04-00-18_0/"

# 获取目录下所有的mp4文件并按数字顺序排序
video_files = sorted([f for f in os.listdir(directory) if f.endswith('.mp4')],
                     key=lambda x: int(re.search(r'\d+', x).group()))

# 输出排序后的视频文件名称
print("排序后的视频文件列表:")
for video in video_files:
    print(video)
    
# 加载所有的视频文件
clips = [VideoFileClip(os.path.join(directory, file)) for file in video_files]

# 将视频合并成一个
final_clip = concatenate_videoclips(clips)

# 导出最终的视频文件
output_file = os.path.join(directory, "merged_video.mp4")
final_clip.write_videofile(output_file, codec="libx264")

print(f"视频合并完成，已保存为: {output_file}")
