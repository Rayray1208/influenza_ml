import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('data.csv')
df2 = pd.read_csv('weekly_temperatures_with_noise.csv')
# 按照共同的列 'YearWeek' 进行合并
# how='inner' 表示取交集，'outer' 表示取并集，'left' 或 'right' 表示保留左或右的所有数据
merged_df = pd.merge(df1, df2, on='YearWeek', how='inner')

# 保存合并后的数据到新文件
merged_df.to_csv('merged_file.csv', index=False)

print(merged_df.head())  # 输出前几行查看结果