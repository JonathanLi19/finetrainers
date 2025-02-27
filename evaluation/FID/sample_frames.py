import os
import shutil
import numpy as np

# 源目录和目标目录
source_dir = '/datadrive3/testset_data/video_as_images'
target_dir = '/datadrive3/testset_data/video_as_images_16frames'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 遍历源目录中的每个子目录
for videoid in os.listdir(source_dir):
    videoid_path = os.path.join(source_dir, videoid)
    if os.path.isdir(videoid_path):
        # 获取子目录中的所有图片文件
        image_files = sorted([f for f in os.listdir(videoid_path) if f.endswith('.jpg') or f.endswith('.png')])

        # 确保有49帧图片
        assert len(image_files) == 49
        # 生成均匀分布的帧索引
        indices = np.linspace(0, 48, 14, dtype=int)

        # 创建目标子目录
        target_videoid_path = os.path.join(target_dir, videoid)
        os.makedirs(target_videoid_path, exist_ok=True)

        # 复制选定的帧到目标子目录
        for idx in indices:
            source_image_path = os.path.join(videoid_path, image_files[idx])
            target_image_path = os.path.join(target_videoid_path, image_files[idx])
            shutil.copy2(source_image_path, target_image_path)

print("Frames have been copied successfully.")