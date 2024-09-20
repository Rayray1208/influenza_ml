import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import wandb
import matplotlib.pyplot as plt

# 初始化 wandb 实验
wandb.init(project="flu_cases_prediction", name="xgboost Model")
allresults = []

def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Args:
    y_true (array-like): Array of actual values
    y_pred (array-like): Array of predicted values
    epsilon (float): Small constant to avoid division by zero
    Returns:
    float: MAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Create a mask for non-zero actual values and non-zero predicted values
    non_zero_mask = (y_true != 0) & (y_pred != 0)  # Ensure both are not zero
    
    # Calculate the absolute percentage error only for non-zero entries
    absolute_percentage_error = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / (y_true[non_zero_mask] + epsilon))
    
    if len(absolute_percentage_error) == 0:
        return np.nan  # Avoid NaN if no data
    
    # Calculate MAPE
    mape = np.mean(absolute_percentage_error) * 100
    
    return mape

def mape_scorer(y_true, y_pred):
    mape = calculate_mape(y_true, y_pred)
    print(f"Current MAPE: {mape:.2f}%")
    allresults.append(mape)
    return -mape  # Return negative MAPE because GridSearchCV tries to maximize score

# 读取数据
data = pd.read_csv("data.csv")

# 数据预处理
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases']]
y = data['ConfirmedCases']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 XGBoost 回归模型
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 定义超参数网格
param_grid = {
    'n_estimators': [500],
    'max_depth': [20],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0.1],
    'min_child_weight': [3]
}

# 创建 GridSearchCV 对象，使用自定义的 MAPE 评分器
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=make_scorer(mape_scorer, greater_is_better=False),  # 寻找最小的 MAPE
    cv=5,
    verbose=2,
    n_jobs=-1
)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳超参数
print("Best parameters found: ", grid_search.best_params_)

# 在 GridSearchCV 完成后记录所有 MAPE 结果到 wandb
for i, mape in enumerate(allresults):
    wandb.log({f"MAPE Fold {i+1}": mape})

# 选择并输出第五个折的 MAPE 结果
if len(allresults) >= 4:
    fifth_fold_mape = allresults[3]  # 0-based index, so 4 is the fifth fold
    print(f"MAPE for the fifth fold: {fifth_fold_mape:.2f}%")
else:
    print("Not enough folds to select the fifth one.")

# 使用最佳超参数的模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = calculate_mape(y_test, y_pred)

# 输出评估结果
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 记录评估结果到 wandb
wandb.log({"MAPE": mape, "MSE": mse, "MAE": mae})

# 可视化实际值与预测值的对比
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
plt.plot(y_pred, label="Predicted", linestyle='--', marker='x')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Confirmed Cases')
plt.title('Actual vs Predicted Confirmed Cases')
plt.savefig('plot_xgboost.png')
plt.show()

# 记录图表到 wandb
wandb.log({"chart": wandb.Image('plot_xgboost.png')})

# 完成 wandb 实验
wandb.finish()