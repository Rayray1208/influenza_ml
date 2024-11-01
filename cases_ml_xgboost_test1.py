import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from xgboost import plot_tree
import wandb
import matplotlib.pyplot as plt

def calculate_mape(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 只排除 y_true 为 0 的情况
    non_zero_mask = y_true != 0
    absolute_percentage_error = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                                       (y_true[non_zero_mask] + epsilon))
    
    # 如果存在有效的非零数据点，计算 MAPE
    return np.mean(absolute_percentage_error) * 100 if len(absolute_percentage_error) > 0 else np.nan
def mape_scorer(y_true, y_pred):
    mape = calculate_mape(y_true, y_pred)
    mape = np.abs(mape) if not np.isnan(mape) else np.inf
    return -mape

def main():
    testname = str(input("intput the test name for this time:"))
    wandb.init(project="model_pre_trained", name = testname)

    data = pd.read_csv("merged_file_with_seasons.csv")
    data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
    data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases',"AverageTemperature",]]
    y = data['ConfirmedCases']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # 原先的 param_grid，假设你基于图片得出该组合
    param_dist ={'colsample_bytree': [1.0], 'gamma': [0.5], 'learning_rate': [0.2], 'max_depth': [15], 'min_child_weight': [5], 'n_estimators': [500], 'reg_alpha': [1], 'reg_lambda': [10], 'subsample': [1.0]}

# 创建 RandomizedSearchCV 对象
    grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(),
        param_grid=param_dist,
        scoring='neg_mean_absolute_percentage_error',  # 使用 MAPE 作为评分标准
        cv =5,  # 5 折交叉验证
        verbose=2,
        n_jobs=-1  # 并行计算
    )

# 执行随机搜索
    grid_search.fit(X_train, y_train)

# 输出最优参数组合
    print("Best parameters found: ", grid_search.best_params_)
    best_params = grid_search.best_params_
    best_mape = -grid_search.best_score_

    # 保存与 70% 最接近的参数组合
    closest_params = None
    closest_mape = float('inf')  # 初始化为无穷大
#####################test
    for params, mean_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        mean_mape = -mean_score  # MAPE 通常是负数形式存储，所以需要取反
    if mean_mape > 0 and mean_mape < closest_mape:  # 确保是正数并且小于当前最小的 mape
        closest_mape = mean_mape
        closest_params = params
##################
    print(f"最接近 0% 的参数组合是: {closest_params}, 对应的 MAPE 是: {closest_mape:.2f}%")
    print("最佳参数组合:", best_params)
    print(f"最佳 MAPE: {best_mape:.2f}%")
    for i, (params, mean_score, std_score) in enumerate(zip(grid_search.cv_results_['params'],
                                                            grid_search.cv_results_['mean_test_score'],
                                                            grid_search.cv_results_['std_test_score'])):
        wandb.log({
            "Fold": i,
            "Mean MAPE": -mean_score,
            "Std MAPE": std_score,
            "Parameters": params
        })
    
    # 用最接近 70% 的参数来训练模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    #best_model.save_model("79xgbooost.model")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)

    print(f"最终模型 MAPE: {mape:.2f}%")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")

    wandb.log({"Final MAPE": mape, "Final MSE": mse, "Final MAE": mae})

    plt.figure(figsize=(10,6))
    plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
    plt.plot(y_pred, label="Predicted", linestyle='--', marker='x')
    plt.legend()
    plt.xlabel('sample index')
    plt.ylabel('comfirmed cases')
    plt.title('Actual vs Predicted comfirmed cases')
    plt.savefig('plot_xgboost(79%).png')
    plt.close()

    plt.figure(figsize=(10, 6))  # Create a new figure
    plot_tree(best_model, num_trees=0)  # 0 表示第一棵树
    plt.title("XGBoost Tree Visualization")
    plt.savefig('xgboost_tree.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure after saving
    plt.show()
    wandb.log({"chart": wandb.Image('plot_xgboost(79%).png')})
    wandb.log({"chart":wandb.Image("xgboost_tree.png")})

if __name__ == "__main__":
    main()
