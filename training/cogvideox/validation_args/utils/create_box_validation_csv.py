import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def bounding_box_extraction(mask_img):
    circle_size = 5
    black_background = np.zeros_like(mask_img)

    # 提取图像中的唯一颜色（排除黑色）
    df = pd.DataFrame(mask_img.reshape(-1, 3), columns=['R', 'G', 'B'])
    unique_colors_df = df.drop_duplicates()
    unique_colors = unique_colors_df.to_numpy()
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]  # 排除黑色
    for color in unique_colors:
        mask = cv2.inRange(mask_img, np.array(color), np.array(color))

        # 获取 mask 的非零坐标点
        coords = np.column_stack(np.where(mask))

        if coords.size > 0:
            # 获取左下角和右下角的坐标
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)

            # 绘制 bounding box，左下角 (min_x, min_y) 和右下角 (max_x, max_y)
            cv2.rectangle(black_background, (min_x, min_y), (max_x, max_y), color.tolist(), circle_size)
    return black_background

# 设置输入输出文件路径
input_csv = 'testset/whole_testset.csv'
output_csv = 'training/cogvideox/validation_args/testset/box/whole_testset.csv'

# 设置保存图片和视频的路径
images_dir = '/datadrive2/lqh/testset_data/images'
box_dir = '/datadrive2/lqh/testset_data/box_condition'

# 读取原始 CSV 文件
df = pd.read_csv(input_csv)

# 创建输出目录，如果不存在
os.makedirs(images_dir, exist_ok=True)
os.makedirs(box_dir, exist_ok=True)


# 处理每一行数据
new_rows = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理数据"):
    videoid = row['videoid']
    text = row['text']
    path = row['path']
    trajectory_maps_path = row['trajectory_maps_path']

    # 生成 trajectory_maps_path 指向的图像视频
    trajectory_maps_files = sorted(os.listdir(trajectory_maps_path))
    video_output_path = os.path.join(box_dir, f"{videoid}.mp4")

    # 如果文件夹内没有图像，跳过视频生成
    if trajectory_maps_files:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 格式
        fps = 30.0  # 假设帧率为30
        first_img = cv2.imread(os.path.join(trajectory_maps_path, trajectory_maps_files[0]))
        height, width, _ = first_img.shape
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

        for i, img_file in enumerate(trajectory_maps_files):
            img_path = os.path.join(trajectory_maps_path, img_file)
            img = cv2.imread(img_path)

            # 提取 bounding box
            if i > 0:
                box_img = bounding_box_extraction(img)
            else:
                box_img = img

            if img is not None:
                out.write(box_img)  # 写入视频
        out.release()

    # 添加新列数据
    new_row = {
        'validation_prompt': text,
        'validation_images': f'/datadrive2/lqh/testset_data/images/{videoid}.jpg',
        'validation_trajectory_maps': f'/datadrive2/lqh/testset_data/box_condition/{videoid}.mp4',
        'output_path': f'samples/box_condition/checkpoint-12900/{videoid}.mp4',
        'controlnet_weights': 1.0
    }
    new_rows.append(new_row)

# 将所有新数据添加到 DataFrame
new_df = pd.DataFrame(new_rows)

# 将结果保存为新的 CSV 文件
new_df.to_csv(output_csv, index=False)

print(f"处理完成，新的 CSV 文件保存在：{output_csv}")