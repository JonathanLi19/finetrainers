import csv
import os
import pandas as pd

# 文件路径
input_csv = 'testset/selected_testset.csv'
mask_fid_csv = 'evaluation/FID/results/mask_condition.csv'
mask_objmc_csv = 'evaluation/ObjMC/results/mask_condition.csv'
box_fid_csv = 'evaluation/FID/results/box_condition.csv'
box_objmc_csv = 'evaluation/ObjMC/results/box_condition.csv'
output_csv = 'testset/final_testset.csv'

# 读取 CSV 文件
selected_testset = pd.read_csv(input_csv)
mask_fid = pd.read_csv(mask_fid_csv).add_suffix('_mask_fid')
mask_objmc = pd.read_csv(mask_objmc_csv).add_suffix('_mask_objmc')
box_fid = pd.read_csv(box_fid_csv).add_suffix('_box_fid')
box_objmc = pd.read_csv(box_objmc_csv).add_suffix('_box_objmc')

# 合并数据并添加后缀
merged = selected_testset.merge(mask_fid, left_on='videoid', right_on='videoid_mask_fid', how='left')
merged = merged.merge(mask_objmc, left_on='videoid', right_on='videoid_mask_objmc', how='left')
merged = merged.merge(box_fid, left_on='videoid', right_on='videoid_box_fid', how='left')
merged = merged.merge(box_objmc, left_on='videoid', right_on='videoid_box_objmc', how='left')

# 删除多余的 videoid 列
merged = merged.drop(columns=['videoid_mask_fid', 'videoid_mask_objmc', 'videoid_box_fid', 'videoid_box_objmc'])

# 过滤掉有 NaN 值的行
merged = merged.dropna()

# 定义一个函数来排序和选择前 100 个视频
def filter_by_score(df, num_objects, n=100):
    if num_objects == '>5':
        df = df[df['num_objects'] > 5]
    else:
        df = df[df['num_objects'] == num_objects]
    df = df.sort_values(by=['fid_value_mask_fid', 'fid_value_box_fid'], ascending=[True, True])
    df = df.sort_values(by=['mask_objectmc_mask_objmc', 'box_objectmc_box_objmc'], ascending=[False, False])
    return df.head(n)

# 过滤每种 num_objects 的视频
filtered = pd.DataFrame()
for num_objects in range(1, 7):
    if num_objects > 5:
        num_objects = '>5'
    filtered = pd.concat([filtered, filter_by_score(merged, num_objects)])

# 保存结果到 CSV 文件
filtered.to_csv(output_csv, index=False)