# This file is used for dataset postprocessing:
# 1. Read video from a csv file
# 2. Crop the video based on masks, ensuring that both the first and last frames have corresponding main-object masks.
# 3. Save the result video
import cv2
import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def is_mask_non_black(mask_frame):
    """Check if a mask frame is non-black (contains any non-zero pixel)."""
    return np.any(mask_frame > 0)

def load_mask_frames(trajectory_maps_path):
    """Load all mask frames from the directory, sorted by filename."""
    mask_filenames = [f for f in os.listdir(trajectory_maps_path) if f.endswith(('.png', '.jpg'))]
    mask_filenames = sorted(mask_filenames)  # Ensure natural sorting (1, 2, 10, etc.)
    
    mask_frames = []
    for mask_filename in mask_filenames:
        mask_path = os.path.join(trajectory_maps_path, mask_filename)
        mask_frame = cv2.imread(mask_path)
        mask_frames.append(mask_frame)
    
    return mask_frames

def process_video(video_path, trajectory_maps_path, output_video_path):
    """Process the video by trimming it based on non-black mask frames."""
    
    # Get sorted mask filenames from the directory
    mask_filenames = [f for f in os.listdir(trajectory_maps_path) if f.endswith(('.png', '.jpg'))]
    mask_filenames = sorted(mask_filenames)  # Ensure natural sorting (1, 2, 10, etc.)
    total_frames = len(mask_filenames)

    if total_frames == 0:
        print(f"No mask frames found in {trajectory_maps_path}")
        return
    
    # Open the video file
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Initialize mask start and end indices
    mask_start_index = None
    mask_end_index = None

    # Find mask_start_index (first non-black mask)
    for i, mask_filename in enumerate(mask_filenames):
        mask_path = os.path.join(trajectory_maps_path, mask_filename)
        mask_frame = cv2.imread(mask_path)

        if is_mask_non_black(mask_frame):
            mask_start_index = i
            break

    # Find mask_end_index (last non-black mask) by iterating backward
    for i in range(total_frames - 1, -1, -1):
        mask_path = os.path.join(trajectory_maps_path, mask_filenames[i])
        mask_frame = cv2.imread(mask_path)

        if is_mask_non_black(mask_frame):
            mask_end_index = i
            break

    # If valid mask indices are not found, exit
    if mask_start_index is None or mask_end_index is None or mask_end_index < mask_start_index:
        print(f"No valid non-black mask frames found.")
        return -1, -1
    # if mask_start_index == 0 and mask_end_index == total_frames - 1:
    #     print(f"No black masks in video {video_path}")
    #     return mask_start_index, mask_end_index

    # print(f"Trimming video from frame {mask_start_index} to frame {mask_end_index}.")

    # # Prepare for output video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fps = video_cap.get(cv2.CAP_PROP_FPS)
    # width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # # Read and write frames from mask_start_index to mask_end_index
    # for i in range(mask_start_index, mask_end_index + 1):
    #     video_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    #     ret, frame = video_cap.read()
    #     if not ret:
    #         break
    #     out_video.write(frame)

    # # Release all resources
    # video_cap.release()
    # out_video.release()

    # print(f"Processed video saved to {output_video_path}")
    return mask_start_index, mask_end_index

def process_row(row, output_path=None):
    video_path = row['path']
    trajectory_maps_path = row['trajectory_maps_path']
    # output_video_path = os.path.join(output_path, os.path.basename(video_path))
    
    # 调用处理视频的函数
    mask_start_index, mask_end_index = process_video(video_path, trajectory_maps_path, output_video_path=None)
    print(f"Finish processing {video_path}")
    
    return row.name, mask_start_index, mask_end_index  # 返回行索引及计算结果

def add_mask_start_end_index(csv_file, output_path=None, max_workers=96):
    """并行读取 CSV 中的视频和 mask 路径，处理视频，并保存索引到 CSV。"""
    start_time = time.time()
    # os.makedirs(output_path, exist_ok=True)
    df = pd.read_csv(csv_file)

    # 创建 mask_start_index 和 mask_end_index 列，如果不存在
    if 'mask_start_index' not in df.columns:
        df['mask_start_index'] = None
    if 'mask_end_index' not in df.columns:
        df['mask_end_index'] = None

    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row, row, output_path=None): row for _, row in df.iterrows()
        }

        for future in as_completed(futures):
            row = futures[future]
            try:
                index, mask_start_index, mask_end_index = future.result()
                # 更新 dataframe 中的 mask 开始/结束索引
                df.at[index, 'mask_start_index'] = mask_start_index
                df.at[index, 'mask_end_index'] = mask_end_index
                df.at[index, 'num_frames'] = mask_end_index - mask_start_index + 1
            except Exception as e:
                print(f"处理行 {row.name} 时出错: {e}")

    # 保存更新后的 dataframe 到 CSV 文件
    df.to_csv(csv_file, index=False)
    print(f"CSV 文件已更新: {csv_file}")
    end_time = time.time()
    print(f"Total time {end_time - start_time} seconds")

if __name__ == '__main__':
    # Path to your CSV file containing 'video_path' and 'mask_path' columns
    csv_file = '/home/qid/quanhao/workspace/Open-Sora/data/Pexels/data_part3.csv'
    add_mask_start_end_index(csv_file)