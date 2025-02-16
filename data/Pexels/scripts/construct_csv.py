import os
import pandas as pd
from PIL import Image

# 从已有的文件构造我们需要的csv文件

def create_csv_from_files():
    video_path = "/datadrive/videos-popular"
    trajectory_maps_path = "/datadrive2/grounded_sam2_output_v100"
    annotations_csv = "/home/qid/quanhao/workspace/Dataset_construction/Sample/Pexels/annotation.csv"
    # 读取注释 CSV 文件
    annotations_df = pd.read_csv(annotations_csv)

    # 创建一个空的列表来存储数据
    data = []

    # 遍历 trajectory_maps_path 目录
    for videoid in os.listdir(trajectory_maps_path):
        videoid_path = os.path.join(trajectory_maps_path, videoid)

        done_file_path = os.path.join(trajectory_maps_path, videoid, "done.txt")

        # 检查 done.txt 文件是否存在
        if not os.path.exists(done_file_path):
            print(f"跳过 videoid: {videoid}，因为 done.txt 文件不存在")
            continue

        # 检查该路径是否是一个目录
        if os.path.isdir(videoid_path):
            # 构造视频路径
            video_file_path = os.path.join(video_path, f"{videoid}.mp4")

            if os.path.exists(video_file_path):
                # 构造 trajectory_maps_path
                masks_images_path = os.path.join(trajectory_maps_path, videoid, "masks_images")

                # 获取一张图片的高度和宽度
                image_files = sorted([f for f in os.listdir(masks_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                # 统计 masks_images_path 中的图片数量
                num_frames = len(image_files)
                if image_files:
                    sample_image_path = os.path.join(masks_images_path, image_files[0])
                    try:
                        with Image.open(sample_image_path) as img:
                            width, height = img.size
                    except Exception as e:
                        print(f"Error processing {sample_image_path}: {e}")
                        # 跳过当前图片，继续处理下一个
                        continue
                else:
                    width, height = None, None  # 如果没有图片，设为 None

                # 从 annotations_df 中获取 name 列信息
                text = annotations_df.loc[annotations_df['videoid'] == int(videoid), 'name'].values
                assert len(text) == 1
                text = text[0] if len(text) > 0 else None  # 获取第一个匹配的值，如果没有则设为 None

                # 将路径和信息添加到数据列表
                data.append({
                    "path": video_file_path,
                    "text": text,
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                    "trajectory_maps_path": masks_images_path,
                })
                print("处理完文件:", video_file_path)
            else:
                print(f"文件不存在，跳过: {video_file_path}")

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 保存为 CSV 文件
    output_csv_path = "/home/qid/quanhao/workspace/Open-Sora/data/Pexels/data_part2.csv"  # 替换为实际输出的 CSV 文件路径
    df.to_csv(output_csv_path, index=False)

    print(f"CSV 文件已保存到 {output_csv_path}")

# 添加videoid到csv文件里面
def add_video_id(csv_file="/home/qid/quanhao/workspace/Open-Sora/data/Pexels/data_part2.csv"):
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 提取 videoid 并添加到新列
    df["videoid"] = df["path"].apply(lambda path: os.path.splitext(os.path.basename(path))[0])
    
    # 保存回原始 CSV 文件
    df.to_csv(csv_file, index=False)
    print(f"Updated CSV file with videoid column saved.")

if __name__ == '__main__':
    create_csv_from_files()
    add_video_id()