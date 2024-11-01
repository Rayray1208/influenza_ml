import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import wandb

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

# Weights & Biases 日志记录函数
def wandb_log(x, y, grid_search=None, mape=None, mse=None, mae=None):
    if x == "n":
        return 0
    else:
        if y == "init":
            proj_name = str(input("input your project name:"))
            wandb.init(project="influenza_formal_test", name=proj_name)
        elif y == "gridsearch" and grid_search is not None:
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
            wandb.log({"chart": wandb.Image('time_series_xgboost.png')})
            wandb.log({"chart": wandb.Image("xgboost_tree.png")})

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
    
    # 初始化 W&B
    use_wandb = input("Do you want to use Weights & Biases for logging? (y/n): ")
    wandb_log(use_wandb, "init")
    
    # 定义网格搜索参数
    param_dist = {
        'n_estimators': [100, 200, 300, 500],  # 树的数量
        'max_depth': [3, 5, 7, 10, 15],  # 树的最大深度
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率
    }
    
    # 创建基础XGBoost模型
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    #Best parameters found:  {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500}
    # 创建GridSearchCV对象
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_dist, 
                               cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    
    # 执行网格搜索
    grid_search.fit(X_train, y_train)
    
    # 记录网格搜索结果
    wandb_log(use_wandb, "gridsearch", grid_search=grid_search)
    
    # 打印最佳参数和分数
    print("Best parameters found: ", grid_search.best_params_)
    print("Best MSE found: ", -grid_search.best_score_)
    
    # 使用最佳参数创建最终模型
    best_model = grid_search.best_estimator_
    
    # 在测试集上进行预测
    y_pred = best_model.predict(X_test)
    
    # 评估结果
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"最终模型 MAPE: {mape:.2f}%")
    
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
    plot_tree(best_model, num_trees=0)  # 0 表示第一棵树
    plt.title("XGBoost Tree Visualization")
    plt.savefig('xgboost_tree.png', dpi=1200, bbox_inches='tight')
    plt.close()  # Close the figure after saving
    
    # 记录最终结果和图表
    wandb_log(use_wandb, "log_final_data", mape=mape, mse=mse, mae=mae)

if __name__ == "__main__":
    main()