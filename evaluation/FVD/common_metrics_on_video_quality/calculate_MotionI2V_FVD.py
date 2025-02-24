import csv
import json
import os

import cv2
import torch
from calculate_fvd import calculate_fvd


# ps: pixel value should be in [0, 1]!

def load_video_to_tensor(video_path, video_tensor, index):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened() and frame_count < VIDEO_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (W, H))
        frame = frame / 255.0  # 将像素值从 [0, 255] 转换到 [0, 1]
        video_tensor[index, frame_count] = torch.tensor(frame).permute(2, 0, 1)
        frame_count += 1
    cap.release()

input_csv = '/datadrive2/lqh/finetrainers/testset/final_testset.csv'
output_csv = 'evaluation/FVD/results/MotionI2V.csv'
VIDEO_LENGTH = 16
CHANNEL = 3
H = 320
W = 512

# 读取 CSV 文件并统计满足条件的视频数量
for num_objects in [1, 2, 3, 4, 5, 6]:
    video_paths = []
    resized_video_paths = []
    with open(input_csv, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if (num_objects == 6 and int(row['num_objects']) > 5) or (int(row['num_objects']) == num_objects):
                video_id = row['videoid']
                output_path = f"/datadrive2/lqh/Motion-I2V/outputs/output_mp4/{video_id}.mp4"
                resized_video_path = f"/datadrive2/lqh/testset_data/resized_videos/{video_id}.mp4"

                if os.path.exists(output_path) and os.path.exists(resized_video_path):
                    video_paths.append(output_path)
                    resized_video_paths.append(resized_video_path)

    NUMBER_OF_VIDEOS = len(video_paths)

    # 初始化视频 Tensor
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, H, W, requires_grad=False)
    videos2 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, H, W, requires_grad=False)

    device = torch.device("cuda")
    # device = torch.device("cpu")

    # 加载视频到 Tensor
    for idx, (output_path, resized_video_path) in enumerate(zip(video_paths, resized_video_paths)):
        print(f"Processing video {idx + 1}...")
        load_video_to_tensor(output_path, videos1, idx)
        load_video_to_tensor(resized_video_path, videos2, idx)

    result = {}
    only_final = True

    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)
    print(json.dumps(result, indent=4))
    fvd_value = result['fvd']['value'][0]
    # 将结果写入 CSV 文件
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['num_objects', 'fvd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 如果文件是空的，写入表头
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'num_objects': num_objects, 'fvd': fvd_value})
