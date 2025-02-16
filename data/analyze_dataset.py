import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 读取CSV文件
def load_data(file_path):
    try:
        # 假设CSV文件的列名是 num_frames, height, width
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


# 统计基本信息
def describe_data(df):
    print("基本统计信息：")
    print(df.describe())  # 输出各列的统计信息（均值、标准差、最小值、最大值等）


# 绘制并保存分布图
def plot_and_save_distributions(df, save_path):
    # 设置绘图风格
    sns.set(style="whitegrid")

    # 创建绘图窗口
    plt.figure(figsize=(12, 4))

    # 绘制 num_frames 的分布图
    plt.subplot(1, 3, 1)
    sns.histplot(df['num_frames'], kde=True, bins=30)
    plt.title('Num Frames Distribution')

    # 绘制 height 的分布图
    plt.subplot(1, 3, 2)
    sns.histplot(df['height'], kde=True, bins=30)
    plt.title('Height Distribution')

    # 绘制 width 的分布图
    plt.subplot(1, 3, 3)
    sns.histplot(df['width'], kde=True, bins=30)
    plt.title('Width Distribution')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path)
    print(f"分布图已保存至: {save_path}")

    # 关闭图像
    plt.close()


# 主函数
if __name__ == "__main__":
    file_path = "/home/qid/quanhao/workspace/Open-Sora/data/Pexels/data.csv"  # 替换为你的CSV文件路径
    save_path = "/home/qid/quanhao/workspace/Open-Sora/data/Pexels/distributions.png"  # 替换为你想要保存图像的路径和文件名

    # 加载数据
    df = load_data(file_path)

    if df is not None:
        # 输出统计信息
        describe_data(df)

        # 绘制并保存分布图
        plot_and_save_distributions(df, save_path)

# 基本统计信息：
#         num_frames       height        width
# count  5916.000000  5916.000000  5916.000000
# mean    586.791920  2709.640297  2039.244760
# std     561.699833  1077.987750   904.979363
# min      32.000000   320.000000   320.000000
# 25%     269.000000  1920.000000  1080.000000
# 50%     417.000000  2160.000000  2160.000000
# 75%     693.000000  3840.000000  2160.000000
# max    7364.000000  4096.000000  4096.000000
