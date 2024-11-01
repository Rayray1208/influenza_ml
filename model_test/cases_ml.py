import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt

# 生成滞后特征的函数
def create_lag_features(data, lag):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['ConfirmedCases'].shift(i)
    return data

# 计算MAPE
def calculate_mape(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = (y_true != 0) & (y_pred != 0)
    absolute_percentage_error = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / (y_true[non_zero_mask] + epsilon))
    return np.mean(absolute_percentage_error) * 100 if len(absolute_percentage_error) > 0 else np.nan

# 主要部分
def main():
    # 读取数据
    data = pd.read_csv("merged_file.csv")
    
    # 创建滞后特征
    data = create_lag_features(data, lag=3)
    
    # 删除包含NaN值的行（因滞后特征生成的NaN）
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

    # 使用 XGBoost 模型
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=500, max_depth=3, learning_rate=0.01)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估结果
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)

    print(f"(MSE): {mse:.2f}")
    print(f"(MAE): {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # 可视化
    plt.figure(figsize=(10,6))
    plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
    plt.plot(y_pred, label="Predicted", linestyle='--', marker='x')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Confirmed Cases')
    plt.title('Actual vs Predicted Confirmed Cases (Time Series)')
    plt.savefig('time_series_xgboost.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))  # Create a new figure
    plot_tree(model, num_trees=0)  # 0 表示第一棵树
    plt.title("XGBoost Tree Visualization")
    plt.savefig('xgboost_tree.png', dpi=1200, bbox_inches='tight')
    plt.close()  # Close the figure after saving

if __name__ == "__main__":
    main()