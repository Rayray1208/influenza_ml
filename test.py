import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 确保 matplotlib 使用中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示问题

# 示例数据
y_test = np.random.rand(10) * 100
y_pred = np.random.rand(10) * 100

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="实际值", linestyle='--', marker='o')
plt.plot(y_pred, label="预测值", linestyle='--', marker='x')
plt.legend()
plt.xlabel('样本序号')
plt.ylabel('确诊病例数')
plt.title('实际值 vs 预测值')
plt.savefig('plot_xgboost.png')
plt.close()