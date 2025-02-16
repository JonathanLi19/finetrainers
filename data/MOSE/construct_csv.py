import json
import csv
import os

# 输入JSON文件路径
json_file_path = '/home/qid/quanhao/workspace/Open-Sora/data/MOSE/meta_train.json'  # 替换为实际的JSON文件路径
csv_file_path = '/home/qid/quanhao/workspace/Open-Sora/data/MOSE/MOSE.csv'  # 输出CSV文件路径

# 读取JSON文件
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 提取视频信息
videos = data['videos']

# 初始化CSV文件
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['path', 'num_frames', 'height', 'width', 'trajectory_maps_path', 'videoid', 'mask_start_index', 'mask_end_index']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 写入CSV头
    writer.writeheader()

    # 遍历每个视频信息并写入CSV
    for videoid, video_info in videos.items():
        # 获取视频的相关信息
        num_frames = video_info['length']
        height = video_info['height']
        width = video_info['width']
        
        # 设置path和trajectory_maps_path
        video_path = os.path.join('/datadrive2/MOSE_release/train/JPEGImages', videoid)
        trajectory_maps_path = os.path.join('/datadrive2/MOSE_release/train/Annotations', videoid)

        mask_start_index = 0
        mask_end_index = num_frames - 1
        
        # 构建行数据并写入CSV
        writer.writerow({
            'path': video_path,
            'num_frames': num_frames,
            'height': height,
            'width': width,
            'trajectory_maps_path': trajectory_maps_path,
            'videoid': videoid,
            'mask_start_index': mask_start_index,
            'mask_end_index': mask_end_index
        })

print(f"CSV file '{csv_file_path}' has been created successfully.")