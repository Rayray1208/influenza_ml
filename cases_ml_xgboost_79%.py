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
    data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
    data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases',"AverageTemperature"]]
    y = data['ConfirmedCases']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # 原先的 param_grid，假设你基于图片得出该组合
    #param_grid = {
   # 'max_depth': [20],#[100,300,500,700,1000]
    #'learning_rate': [0.1],
    #'subsample': [0.8],
    #'colsample_bytree': [0.8],
    #'gamma': [0.1],
    #'min_child_weight': [3]
    #}
    ###
    param_grid = {
        'reg_lambda': [1], 
        'reg_alpha': [0],
        'n_estimators': [300], 
        'min_child_weight': [1], 
        'max_depth': [15], 
        'learning_rate': [0.05], 
        'gamma': [0.1]
}
    ###

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(mape_scorer, greater_is_better=False),
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_mape = -grid_search.best_score_

    # 保存与 70% 最接近的参数组合
    closest_params = None
    closest_mape = float('inf')  # 初始化为无穷大

    for params, mean_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        mean_mape = -mean_score
        if abs(mean_mape - 40) < abs(closest_mape - 40):  # 找到与 70% 最接近的
            closest_mape = mean_mape
            closest_params = params

    print(f"最接近 0% 的参数组合是: {closest_params}, 对应的 MAPE 是: {closest_mape:.2f}%")
    print("最佳参数组合:", best_params)
    print(f"最佳 MAPE: {best_mape:.2f}%")
    wandb_log(log1,"grid_search")
    
    # 用最接近 70% 的参数来训练模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_model.save_model("79%_xgboost.ubj")
    best_model.save_model("79%_xgboost.model")

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
    plot_tree(best_model, num_trees=0,fontsize = 12)  # 0 表示第一棵树
    plt.title("XGBoost Tree Visualization")
    plt.savefig("xgboost_tree.png",dpi = 1200 , bbox_inches = "tight")
    plt.close()  # Close the figure after saving
    wandb_log(log1,"log_final_data")
if __name__ == "__main__":
    main()