import pandas as pd
for num_objects in [0, 1, 2, 3, 4, 5, 6]:

    # 文件路径
    final_testset_path = 'testset/final_testset.csv'
    mask_condition_final_path = 'evaluation/FID/results/DragNUWA.csv'

    # 读取 CSV 文件
    final_testset_df = pd.read_csv(final_testset_path)
    mask_condition_final_df = pd.read_csv(mask_condition_final_path)

    # 过滤 num_objects=1 的行
    if num_objects == 0:
        filtered_final_testset_df = final_testset_df
    elif num_objects == 6:
        filtered_final_testset_df = final_testset_df[final_testset_df['num_objects'] > 5]
    else:
        filtered_final_testset_df = final_testset_df[final_testset_df['num_objects'] == num_objects]

    # 获取 num_objects=1 对应的 videoid 列表
    videoid_list = filtered_final_testset_df['videoid'].tolist()

    # 过滤 mask_condition_final_df 中 videoid 在 videoid_list 中的行
    filtered_mask_condition_final_df = mask_condition_final_df[mask_condition_final_df['videoid'].isin(videoid_list)]

    # 计算 'fid_value' 列的平均值
    average_fid = filtered_mask_condition_final_df['fid_value'].mean()

    # 输出结果
    print(f'{num_objects} 个物体的fid_value 平均值是: {average_fid}')

# DragAnything: 146.3747869303462
# Tora: 144.4946397723322
# box_condition: 93.27074045549811
# mask_condition: 87.12836356674016
# LeViTor: 187.40314096931476
