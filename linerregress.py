import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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
    wandb.init(project="influenza_formal_test", name="linear_regression_time_series_model")

    data = pd.read_csv("merged_file.csv")
    data = create_lag_features(data, lag=3)
    
    # 删除包含NaN值的行（因滞后特征生成的NaN）
    data.dropna(inplace=True)

    data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
    data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', "AverageTemperature", 'lag_1', 'lag_2', 'lag_3']]
    y = data['ConfirmedCases']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义要测试的模型和它们的参数网格
    models = {
        'Linear Regression': (LinearRegression(), {'fit_intercept': [True, False]}),
        'Ridge Regression': (Ridge(), {'alpha': [0.1, 1.0, 10.0], 'fit_intercept': [True, False]}),
        'Lasso Regression': (Lasso(), {'alpha': [0.1, 1.0, 10.0], 'fit_intercept': [True, False]}),
        'ElasticNet': (ElasticNet(), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8], 'fit_intercept': [True, False]})
    }

    best_model = None
    best_mape = float('inf')

    for model_name, (model, param_grid) in models.items():
        print(f"\n正在训练 {model_name}...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=make_scorer(mape_scorer, greater_is_better=False),
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        current_mape = -grid_search.best_score_
        print(f"{model_name} 最佳参数: {grid_search.best_params_}")
        print(f"{model_name} MAPE: {current_mape:.2f}%")
        
        for i, (params, mean_score, std_score) in enumerate(zip(grid_search.cv_results_['params'],
                                                                grid_search.cv_results_['mean_test_score'],
                                                                grid_search.cv_results_['std_test_score'])):
            wandb.log({
                "Model": model_name,
                "Fold": i,
                "Mean MAPE": -mean_score,
                "Std MAPE": std_score,
                "Parameters": params
            })
        
        if current_mape < best_mape:
            best_mape = current_mape
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    print(f"\n最佳模型: {best_model_name}")
    print(f"最佳 MAPE: {best_mape:.2f}%")

    # 使用最佳模型进行预测
    y_pred = best_model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)

    print(f"最终模型 MAPE: {mape:.2f}%")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")

    wandb.log({"Final Model": best_model_name, "Final MAPE": mape, "Final MSE": mse, "Final MAE": mae})

    plt.figure(figsize=(10,6))
    plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
    plt.plot(y_pred, label="Predicted", linestyle='--', marker='x')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Confirmed Cases')
    plt.title(f'Actual vs Predicted Confirmed Cases ({best_model_name})')
    plt.savefig('plot_linear_regression.png')
    plt.close()

    wandb.log({"chart": wandb.Image('plot_linear_regression.png')})

if __name__ == "__main__":
    main()