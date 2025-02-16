import pandas as pd

# 读取 CSV 文件
csv_file_path = "/home/qid/quanhao/workspace/Open-Sora/data/MOSE/MOSE.csv"
df = pd.read_csv(csv_file_path)

# 删除原有的 "text" 列（如存在）
if "text" in df.columns:
    df = df.drop(columns=["text"])

# 插入新列 "text"，内容为两个英文字符
df.insert(0, "text", "")  # 插入两个英文双引号

# 保存文件
df.to_csv(csv_file_path, index=False)

# 重新读取文件
df_reloaded = pd.read_csv(csv_file_path)
print(df_reloaded["text"].head())  # 验证内容