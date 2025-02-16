import csv

# 输入CSV文件路径
csv_file_path = '/home/qid/quanhao/workspace/Open-Sora/data/MOSE/MOSE.csv'  # 替换为实际的CSV文件路径

# 统计 num_frames >= 51 且 width * height >= 409920 的视频数量
count = 0

# 打开CSV文件并读取
with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # 遍历CSV文件中的每一行
    for row in reader:
        # 获取num_frames, width 和 height 字段并转换为整数
        num_frames = int(row['num_frames'])
        width = int(row['width'])
        height = int(row['height'])
        
        # 判断 num_frames 是否大于或等于 51 且 width * height 是否大于或等于 409920
        if num_frames >= 51 and width * height >= 409920:
            count += 1

print(f"There are {count} videos with num_frames >= 51 and width * height >= 409920.")