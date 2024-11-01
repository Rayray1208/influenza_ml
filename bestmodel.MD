import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost import plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import wandb
import matplotlib.pyplot as plt
def wandb_log(x,y):
    if x == "n":
        return 0 
    else :
        if y == "init":
            proj_name = str(input("input your project name:"))
            wandb.init(project="influenza_formal_test", name=proj_name)
        elif y == "gridsearch":
            for i, (params, mean_score, std_score) in enumerate(zip(grid_search.cv_results_['params'],
                                                            grid_search.cv_results_['mean_test_score'],
                                                            grid_search.cv_results_['std_test_score'])):
                wandb.log({
                 "Fold": i,
                "Mean MAPE": -mean_score,
                "Std MAPE": std_score,
                "Parameters": params
        })
        elif y == "log_final_data" and mape is not None and mse is not None and mae is not None:
            wandb.log({"Final MAPE": mape, "Final MSE": mse, "Final MAE": mae})
            wandb.log({"chart": wandb.Image('plot_xgboost.png')})
            wandb.log({"chart": wandb.Image("xgboost_tree.png")})
def create_lag_features(data, lag):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['ConfirmedCases'].shift(i)
    return data
        
def calculate_mape(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = (y_true != 0) & (y_pred != 0)
    absolute_percentage_error = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / (y_true[non_zero_mask] + epsilon))
    return np.mean(absolute_percentage_error) * 100 if len(absolute_percentage_error) > 0 else np.nan

def mape_scorer(y_true, y_pred):
    mape = calculate_mape(y_true, y_pred)
    mape = np.abs(mape) if not np.isnan(mape) else np.inf
    return -mape

def main():
    log1 = str(input("log to wandb y/n?"))
    wandb_log(log1,"init")

    data = pd.read_csv("merged_file.csv")
    

    gdata = create_lag_features(data, lag=3)
    
    # 删除包含NaN值的行（因滞后特征生成的NaN）
    data.dropna(inplace=True)
    
    # 提取年份和周数
    data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
    data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

    # 特征包括年份、周数、滞后特征、温度等
    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', 'AverageTemperature', 'lag_1', 'lag_2', 'lag_3']]
    y = data['ConfirmedCases']

    # 按时间顺序分割数据，80%用于训练，20%用于测试
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 使用 XGBoost 模型
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=300, max_depth=10, learning_rate=0.05)

    # 训练模型
    best_nmodel = 
    best_model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)

    print(f"最终模型 MAPE: {mape:.2f}%")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")

    plt.figure(figsize=(10,6))
    plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
    plt.plot(y_pred, label="Predicted", linestyle='--', marker='x')
    plt.legend()
    plt.xlabel('sample index')
    plt.ylabel('comfirmed cases')
    plt.title('Actual vs Predicted comfirmed cases')
    plt.savefig('plot_xgboost.png')
    plt.close()

    plt.figure(figsize=(20,12))
    plot_tree(model, num_trees=0,fontsize = 12)  # 0 表示第一棵树
    plt.title("XGBoost Tree Visualization")
    plt.savefig("xgboost_tree.png",dpi = 1200 , bbox_inches = "tight")
    plt.close()  # Close the figure after saving
    wandb_log(log1,"log_final_data")
if __name__ == "__main__":
     model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
