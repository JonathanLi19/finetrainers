import pandas as pd

# 读取 selected_testset.csv 中的 videoid 列，并转换为整数类型
selected_testset_df = pd.read_csv('testset/selected_testset.csv')
selected_testset_df['videoid'] = selected_testset_df['videoid'].astype(int)
selected_videoids = selected_testset_df['videoid'].unique()

# 读取 testset.csv 文件
testset_df = pd.read_csv('training/cogvideox/validation_args/testset/mask/whole_testset.csv')

# 提取 validation_images 列中的 videoid（通过路径截取），并转换为整数类型
testset_df['videoid_from_path'] = testset_df['validation_images'].str.extract(r'/(\d+)\.jpg$', expand=False).astype(int)

# 筛选出 videoid 在 selected_testset_df 中的行
filtered_df = testset_df[testset_df['videoid_from_path'].isin(selected_videoids)]

# 去重，只保留每个 videoid 一次
filtered_df = filtered_df.drop_duplicates(subset='videoid_from_path')

# 保存结果为新的 CSV 文件
filtered_df.to_csv('training/cogvideox/validation_args/testset/mask/selected_testset.csv', index=False)

print(f"筛选后的数据已保存到 training/cogvideox/validation_args/testset/mask/selected_testset.csv")