import pandas as pd

# 读取 CSV 文件
file_path = 'evaluation/FID/results/LeviTor.csv'
df = pd.read_csv(file_path)

# 计算 'fid_value' 列的平均值
average_fid = df['fid_value'].mean()

# 输出结果
print(f'FID列的平均值是: {average_fid}')

# DragAnything: 146.3747869303462
# Tora: 144.4946397723322
# box_condition: 114.31806044737142
# mask_condition: 105.64298297478429