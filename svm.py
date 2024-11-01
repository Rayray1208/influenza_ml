import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
import wandb
import matplotlib.pyplot as plt
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
    wandb.init(project="influenza_formal_test", name="svm_model_time_series_test1")
    
    data = pd.read_csv("merged_file.csv")
    gdata = create_lag_features(data, lag=3)
    
    # 删除包含NaN值的行（因滞后特征生成的NaN）
    data.dropna(inplace=True)
    
    data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
    data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)
    
    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', "AverageTemperature"]]
    y = data['ConfirmedCases']
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVR()
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.1, 1],
        'epsilon': [0.1, 0.2, 0.5]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(mape_scorer, greater_is_better=False),
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_params = grid_search.best_params_
    best_mape = -grid_search.best_score_
    
    # 寻找最接近 40% MAPE 的参数组合
    closest_params = None
    closest_mape = float('inf')
    for params, mean_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        mean_mape = -mean_score
        if abs(mean_mape - 40) < abs(closest_mape - 40):
            closest_mape = mean_mape
            closest_params = params
    
    print(f"最接近 40% 的参数组合是: {closest_params}, 对应的 MAPE 是: {closest_mape:.2f}%")
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
    
    # 使用最佳参数训练模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
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
    plt.ylabel('confirmed cases')
    plt.title('Actual vs Predicted confirmed cases (SVM)')
    plt.savefig('plot_svm.png')
    plt.close()
    
    wandb.log({"chart": wandb.Image('plot_svm.png')})

if __name__ == "__main__":
    main()