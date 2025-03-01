import os
import cv2
import pandas as pd
from tqdm import tqdm

# 设置输入输出路径
input_csv = 'data/DAVIS/DAVIS_data.csv'
output_csv = 'training/cogvideox/validation_args/testset.csv'

# 设置保存图片和视频的路径
images_dir = '/datadrive2/lqh/testset_data/images'
masks_dir = '/datadrive2/lqh/testset_data/masks_condition'

# 读取原始 CSV 文件
df = pd.read_csv(input_csv)

# 创建输出目录，如果不存在
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# 处理每一行数据
new_rows = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理数据"):
    videoid = row['videoid']
    text = row['text']
    path = row['path']
    trajectory_maps_path = row['trajectory_maps_path']
    
    # 提取视频的第一帧并保存为图像
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()  # 读取第一帧
    if ret:
        first_frame_path = os.path.join(images_dir, f"{videoid}.jpg")
        cv2.imwrite(first_frame_path, frame)
    cap.release()
    
    # 生成 trajectory_maps_path 指向的图像视频
    trajectory_maps_files = sorted(os.listdir(trajectory_maps_path))
    video_output_path = os.path.join(masks_dir, f"{videoid}.mp4")
    
    # 如果文件夹内没有图像，跳过视频生成
    if trajectory_maps_files:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 格式
        fps = 30.0  # 假设帧率为30
        first_img = cv2.imread(os.path.join(trajectory_maps_path, trajectory_maps_files[0]))
        height, width, _ = first_img.shape
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        for img_file in trajectory_maps_files:
            img_path = os.path.join(trajectory_maps_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                out.write(img)  # 写入视频
        
        out.release()

    # 添加新列数据
    new_row = {
        'validation_prompt': text,
        'validation_images': f'/datadrive2/lqh/testset_data/images/{videoid}.jpg',
        'validation_trajectory_maps': f'/datadrive2/lqh/testset_data/masks_condition/{videoid}.mp4',
        'output_path': f'samples/mask_condition/checkpoint-17000/{videoid}.mp4',
        'controlnet_weights': 1.0
    }
    new_rows.append(new_row)

# 将所有新数据添加到 DataFrame
new_df = pd.DataFrame(new_rows)

# 将结果保存为新的 CSV 文件
new_df.to_csv(output_csv, index=False)

print(f"处理完成，新的 CSV 文件保存在：{output_csv}")