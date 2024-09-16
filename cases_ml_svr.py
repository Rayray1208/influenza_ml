import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import wandb

# 初始化 wandb 项目
wandb.init(project="flu_cases_prediction", name="SVR Model")

# 读取病例数据

data = pd.read_csv("data.csv")
# 将 YearWeek 转换为年和周两个特征
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

# 选择特征和目标变量（这里以确诊病例数为目标）
X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases']]
y = data['ConfirmedCases']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建并训练 SVR 模型
model = SVR(kernel='rbf')  # 使用径向基函数核
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)

# 计算 MAPE
non_zero_mask = y_test != 0
y_true_non_zero = y_test[non_zero_mask]
y_pred_non_zero = y_pred[non_zero_mask]

# 确保非零目标值不为空
if len(y_true_non_zero) > 0:
    mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
else:
    mape = float('inf')

wandb.log({"MAPE": mape})  # 将 MAPE 上传至 wandb

# 打印 MAPE 值
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 可视化实际值与预测值的对比
plt.figure(figsize=(10,6))
plt.plot(y_test.reset_index(drop=True).values, label="Actual")
plt.plot(pd.Series(y_pred).reset_index(drop=True).values, label="Predicted")
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Confirmed Cases')
plt.title('Actual vs Predicted Confirmed Cases')

# 保存图像并上传至 wandb
plt.savefig('plot_svr.png')
plt.show
wandb.log({"chart": wandb.Image('plot_svr.png')})

# 完成 wandb 实验
wandb.finish()