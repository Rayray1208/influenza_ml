import pandas as pd
import numpy as np

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 每个月的平均温度数据 (以淡水为例)
monthly_temperatures = {
    'January': 15.4, 'February': 15.7, 'March': 17.7, 'April': 21.4,
    'May': 24.7, 'June': 27.3, 'July': 29.0, 'August': 28.7,
    'September': 26.9, 'October': 23.6, 'November': 21.0, 'December': 17.3
}

# 将月份的平均气温按顺序存入列表
monthly_temps = list(monthly_temperatures.values())

# 创建52周的日期范围（从2010年第1周到2020年第17周）
weeks = pd.date_range('2010-01-01', periods=522, freq='W-MON')

# 使用线性插值生成每周的基础平均温度
weekly_temperatures = np.interp(np.linspace(0, 12, len(weeks)), range(12), monthly_temps)

# 添加一个小范围的随机波动，保持波动在 ±1°C 左右
random_fluctuations = np.random.normal(loc=0, scale=1, size=len(weeks))

# 生成加入波动后的每周平均温度
weekly_temperatures_with_noise = weekly_temperatures + random_fluctuations

# 将日期和温度整合到DataFrame中
df = pd.DataFrame({
    'YearWeek': weeks.strftime('%Y%W'),
    'AverageTemperature': weekly_temperatures_with_noise
})

# 导出为CSV文件
df.to_csv('weekly_temperatures_with_noise.csv', index=False)

print("带有随机波动的 CSV 文件已生成")