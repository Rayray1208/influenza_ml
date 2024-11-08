import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 增加 LSTM 模型：在 XGBoost 的預測後，增加了 LSTM 的建立和訓練過程，這樣可以比較兩種模型的預測效果。
# 數據標準化：使用 MinMaxScaler 對 ConfirmedCases 進行標準化，以便於 LSTM 的訓練。
# 創建資料集：用 create_dataset 函數來為 LSTM 準備訓練和測試資料。
# 模型評估：分別計算 XGBoost 和 LSTM 的 MSE、MAE 和 MAPE，方便後續比較。
# 可視化：在同一圖中顯示 XGBoost 和 LSTM 的預測結果，方便比較。

# 生成滞后特征的函数
def create_lag_features(data, lag):
    # 根據滯後天數生成新的滯後特徵
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['ConfirmedCases'].shift(i)
    return data

# 计算MAPE
def calculate_mape(y_true, y_pred, epsilon=1e-10):
    # 計算平均絕對百分比誤差 (MAPE)
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

    # 特征和目标变量
    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', 'AverageTemperature', 'lag_1', 'lag_2', 'lag_3']]
    y = data['ConfirmedCases']

    # 按时间顺序分割数据，80%用于训练，20%用于测试
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 使用 XGBoost 模型
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=500, max_depth=3, learning_rate=0.01)

    # 训练模型
    model_xgb.fit(X_train, y_train)

    # 预测
    y_pred_xgb = model_xgb.predict(X_test)

    # 评估结果
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mape_xgb = calculate_mape(y_test, y_pred_xgb)

    # 输出 XGBoost 的评估结果
    print(f"XGBoost (MSE): {mse_xgb:.2f}")
    print(f"XGBoost (MAE): {mae_xgb:.2f}")
    print(f"XGBoost MAPE: {mape_xgb:.2f}%")

    # LSTM 模型部分
    # 數據標準化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['ConfirmedCases']])

    # 創建 LSTM 的資料格式
    def create_dataset(dataset, time_step=1):
        # 將資料轉換為 LSTM 所需的格式
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    # 設定時間步長
    time_step = 3
    X_lstm, y_lstm = create_dataset(scaled_data, time_step)

    # 重新形狀以符合 LSTM 的輸入要求
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    # 構建 LSTM 模型
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1))

    # 編譯模型
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # 訓練模型
    model_lstm.fit(X_lstm, y_lstm, epochs=100, batch_size=32)

    # 預測
    y_pred_lstm = model_lstm.predict(X_lstm)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

    # 評估 LSTM 的結果
    mse_lstm = mean_squared_error(y_lstm, y_pred_lstm)
    mae_lstm = mean_absolute_error(y_lstm, y_pred_lstm)
    mape_lstm = calculate_mape(y_lstm, y_pred_lstm)

    # 输出 LSTM 的评估结果
    print(f"LSTM (MSE): {mse_lstm:.2f}")
    print(f"LSTM (MAE): {mae_lstm:.2f}")
    print(f"LSTM MAPE: {mape_lstm:.2f}%")

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="XGBoost Actual", linestyle='--', marker='o')
    plt.plot(y_pred_xgb, label="XGBoost Predicted", linestyle='--', marker='x')
    plt.plot(y_lstm, label="LSTM Actual", linestyle='--', marker='o', alpha=0.5)
    plt.plot(y_pred_lstm, label="LSTM Predicted", linestyle='--', marker='x', alpha=0.5)
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Confirmed Cases')
    plt.title('Actual vs Predicted Confirmed Cases (XGBoost vs LSTM)')
    plt.savefig('actual_vs_predicted.png', dpi=300)
    plt.show()

    # 可视化 XGBoost 树
    plt.figure(figsize=(10, 6))  # Create a new figure
    plot_tree(model_xgb, num_trees=0)  # 0 表示第一棵树
    plt.title("XGBoost Tree Visualization")
    plt.savefig('xgboost_tree.png', dpi=1200, bbox_inches='tight')
    plt.close()  # Close the figure after saving

if __name__ == "__main__":
    main()
