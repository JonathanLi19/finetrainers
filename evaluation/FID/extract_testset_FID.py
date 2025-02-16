import os
import subprocess

# 设置根路径
root_path = '/datadrive2/lqh/testset_data/video_as_images'

# 遍历根路径下的每个子目录
for subdir in os.listdir(root_path):
    subdir_path = os.path.join(root_path, subdir)

    # 检查是否是目录
    if os.path.isdir(subdir_path):
        # 拼接保存路径
        output_file = os.path.join(subdir_path, f'{subdir}.npz')

        # 检查文件是否已经存在
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping.")
        else:
            # 拼接命令
            command = f"python -m pytorch_fid --save-stats {subdir_path} {output_file}"

            # 执行命令
            subprocess.run(command, shell=True, check=True)
            print(f"Executed: {command}")