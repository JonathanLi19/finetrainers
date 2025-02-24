import csv
import os
import re
import subprocess

import cv2


# 读取原始 CSV 文件中的 videoid 列
input_csv = 'testset/final_testset.csv'
output_csv = 'evaluation/FID/results/MotionI2V.csv'

# 存储 FID 值的结果
fid_results = []

# 创建保存图像的函数
def extract_frames(video_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 保存每一帧图像
        frame_filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

# 执行 FID 计算命令
def calculate_fid(video_id):
    command = [
        "python",
        "-m",
        "pytorch_fid",
        f"/datadrive2/lqh/generated_videos_as_images/MotionI2V/{video_id}",
        f"/datadrive2/lqh/testset_data/video_as_images/{video_id}/{video_id}.npz"
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # 提取 FID 值
    match = re.search(r"FID:\s+([\d\.]+)", result.stdout)
    if match:
        return float(match.group(1))
    else:
        return None

# 读取已存在的 FID 结果
existing_fid_results = {}
if os.path.exists(output_csv):
    with open(output_csv, 'r') as outfile:
        reader = csv.DictReader(outfile)
        for row in reader:
            existing_fid_results[row['videoid']] = float(row['fid_value'])

# 处理 CSV 文件
with open(input_csv, 'r') as infile:
    reader = csv.DictReader(infile)

    for row in reader:
        video_id = row['videoid']
        output_path = f"/datadrive2/lqh/Motion-I2V/outputs/output_mp4/{video_id}.mp4"

        # 检查视频文件是否存在以及是否已经处理过
        if os.path.exists(output_path) and video_id not in existing_fid_results:
            save_dir = f"/datadrive2/lqh/generated_videos_as_images/MotionI2V/{video_id}"

            if not os.path.exists(save_dir):
                # 提取视频的每一帧
                extract_frames(output_path, save_dir)

            # 计算 FID 值
            fid_value = calculate_fid(video_id)
            print(f"Video {video_id}: FID = {fid_value}")

            # 如果 FID 值有效，立即保存到 CSV 文件
            if fid_value is not None:
                with open(output_csv, 'a' if os.path.exists(output_csv) else 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    if os.path.getsize(output_csv) == 0:
                        writer.writerow(['videoid', 'fid_value'])  # 写入表头
                    writer.writerow([video_id, fid_value])

print(f"FID values have been saved to {output_csv}")
