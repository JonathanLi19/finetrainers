# import json

# # 读取 JSON 文件并计算符合条件的视频个数
# def count_videos_with_min_length(json_path, min_length=51):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     videos = data.get("videos", {})
#     count = sum(1 for video in videos.values() if len(video.get("frames", [])) >= min_length)
    
#     print(f"Number of videos with length >= {min_length}: {count}")
#     return count # 888

# json_path = "data/MeViS/meta_expressions.json"
# count_videos_with_min_length(json_path)

import pandas as pd

# 读取 CSV 文件并计算符合条件的视频个数
def count_videos_with_min_length(csv_path, min_length=51):
    df = pd.read_csv(csv_path)
    
    count = (df['num_frames'] > min_length).sum()
    
    print(f"Number of videos with num_frames > {min_length}: {count}")
    return count

csv_path = "data/MeViS/MeViS.csv"
count_videos_with_min_length(csv_path)