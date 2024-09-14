import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import wandb

# 初始化 wandb 项目
wandb.init(project="flu_cases_prediction", name="Ridge_Regression_Model")

# 读取病例数据
data = pd.read_csv("data.csv")

# 将 YearWeek 转换为年和周两个特征
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

# 选择特征和目标变量（这里以确诊病例数为目标）
X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases']]
y = data['ConfirmedCases']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建 Ridge 回归模型
random_seed = 42
model = Ridge(random_state=random_seed)

# 定义超参数网格
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳超参数
print("Best Parameters:", grid_search.best_params_)

# 使用最佳超参数的模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 计算模型评估指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 记录模型的性能
wandb.log({"MSE": mse, "MAE": mae})

# 打印评估指标
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 可视化实际值与预测值的对比
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
plt.plot(y_pred, label="Predicted", linestyle='-', marker='x')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Confirmed Cases')
plt.title('Actual vs Predicted Confirmed Cases')
plt.show()

# 将结果上传至 wandb
wandb.log({"chart": plt})

# 完成 wandb 实验
wandb.finish()