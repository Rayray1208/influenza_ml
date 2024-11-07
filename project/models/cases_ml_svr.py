import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import wandb

def create_lag_features(data, lag):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['ConfirmedCases'].shift(i)
    return data

# 自定义MAPE评分器
def mape(y_true, y_pred):
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# 初始化 wandb 项目
wandb.init(project="influenza_formal_test", name="SVR Model with Grid Search and Metrics Logging")

# 询问是否需要记录到 wandb
record_wandb = input("是否记录 wandb 数据？(y/n): ").strip().lower() == 'y'

# 读取病例数据
data = pd.read_csv("merged_file.csv")

# 创建滞后特征
data = create_lag_features(data, lag=3)
data.dropna(inplace=True)

# 提取年份和周数
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', 'AverageTemperature', 'lag_1', 'lag_2', 'lag_3']]
y = data['ConfirmedCases']

# 按时间顺序分割数据，80%用于训练，20%用于测试
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 定义参数网格
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 0.0001, 0.001, 0.01, 0.1, 1, 10],
    'epsilon': [0.001, 0.01, 0.1, 1, 5],
    'kernel': ['rbf']
}

# 定义评分器
scoring = {
    'MAPE': make_scorer(mape, greater_is_better=False),
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
    'MAE': make_scorer(mean_absolute_error, greater_is_better=False)
}

# 使用 GridSearchCV 进行网格搜索
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring=scoring, refit='MAPE', return_train_score=True, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数和得分
best_params = grid_search.best_params_
best_mape = -grid_search.best_score_
print("Best parameters:", best_params)
print("Best MAPE score from Grid Search:", best_mape)

# 记录每次搜索的结果（根据选择记录到 wandb）
if record_wandb:
    for i, params in enumerate(grid_search.cv_results_['params']):
        mape_score = -grid_search.cv_results_['mean_test_MAPE'][i]
        mse_score = -grid_search.cv_results_['mean_test_MSE'][i]
        mae_score = -grid_search.cv_results_['mean_test_MAE'][i]
        wandb.log({
            "MAPE": mape_score,
            "MSE": mse_score,
            "MAE": mae_score,
            "Parameters": params
        })

# 使用最佳参数的模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 计算最终的 MAPE、MSE 和 MAE
final_mape = mape(y_test, y_pred)
final_mse = mean_squared_error(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)

# 记录最终结果（根据选择记录到 wandb）
if record_wandb:
    wandb.log({
        "Best Parameters": best_params,
        "Final MAPE": final_mape,
        "Final MSE": final_mse,
        "Final MAE": final_mae
    })

# 打印最终MAPE值
print(f"Final MAPE: {final_mape:.2f}%")
print(f"Final MSE: {final_mse:.2f}")
print(f"Final MAE: {final_mae:.2f}")

# 可视化实际值与预测值的对比
plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True).values, label="Actual", linestyle='--', marker='o')
plt.plot(pd.Series(y_pred).reset_index(drop=True).values, label="Predicted", linestyle='--', marker='x')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Confirmed Cases')
plt.title('Actual vs Predicted Confirmed Cases with Grid Search')
plt.show()

# 保存图像并上传至 wandb（根据选择记录到 wandb）
if record_wandb:
    plt.savefig('plot_svr_grid_search.png')
    wandb.log({"chart": wandb.Image('plot_svr_grid_search.png')})

# 完成 wandb 实验
if record_wandb:
    wandb.finish()