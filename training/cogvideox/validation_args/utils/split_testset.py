import pandas as pd
import os

# 读取 CSV 文件
input_file = '/home/qid/quanhao/workspace/finetrainers/training/cogvideox/validation_args/testset/box/final_testset.csv'
df = pd.read_csv(input_file)

# 计算每个部分的大小
num_parts = 4
part_size = len(df) // num_parts

# 创建输出目录
output_dir = '/home/qid/quanhao/workspace/finetrainers/training/cogvideox/validation_args/testset/box/local_part'
os.makedirs(output_dir, exist_ok=True)

# 分割并保存 CSV 文件
for i in range(num_parts):
    start_idx = i * part_size
    end_idx = (i + 1) * part_size if i != num_parts - 1 else len(df)
    part_df = df.iloc[start_idx:end_idx]
    output_file = os.path.join(output_dir, f'part_{i + 1}.csv')
    part_df.to_csv(output_file, index=False)

print("CSV 文件已成功分割并保存。")