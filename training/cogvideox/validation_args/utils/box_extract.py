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

def process_videos():
    jpeg_images_dir = 'data/DAVIS/Annotations/480p'
    output_dir = 'testset/box_condition/DAVIS'
    os.makedirs(output_dir, exist_ok=True)

    for video_id in os.listdir(jpeg_images_dir):
        video_dir = os.path.join(jpeg_images_dir, video_id)
        if os.path.isdir(video_dir):
            frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.png')])[:49]
            if not frames:
                continue

            # 获取第一帧作为mask
            first_frame_path = os.path.join(video_dir, frames[0])
            mask_img = cv2.imread(first_frame_path)
            height, width, _ = mask_img.shape

            # 创建视频写入对象
            output_video_path = os.path.join(output_dir, f'{video_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

            for frame in frames:
                frame_path = os.path.join(video_dir, frame)
                img = cv2.imread(frame_path)
                if frame == frames[0]:
                    processed_img = mask_img
                else:
                    processed_img = bounding_box_extraction(img)
                video_writer.write(processed_img)

            video_writer.release()
            print(f'Saved video {output_video_path}')

if __name__ == '__main__':
    process_videos()