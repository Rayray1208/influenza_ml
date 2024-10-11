import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import wandb

# 初始化 wandb 项目
wandb.init(project="flu_cases_prediction", name="Random Forest Model")

# 读取病例数据
data = pd.read_csv("data.csv")

# 将 YearWeek 转换为年和周两个特征
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

# 选择特征和目标变量
X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases']]
y = data['ConfirmedCases']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建 RandomForestRegressor 模型
model = RandomForestRegressor(random_state=42)

# 定义超参数网格
n_estimators_range = [100, 200, 500, 1000]
max_depth_range = [None, 10, 20]
min_samples_split_range = [2, 5, 10]
min_samples_leaf_range = [1, 2, 5, 10]

# 创建 GridSearchCV 对象
#grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# 训练模型
#grid_search.fit(X_train, y_train)

# 输出最佳超参数
#print("Best Parameters:", grid_search.best_params_)

# 使用最佳超参数的模型进行预测
#best_model = grid_search.best_estimator_
#y_pred = best_model.predict(X_test)


# 计算 MAPE
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        for min_samples_split in min_samples_split_range:
            for min_samples_leaf in min_samples_leaf_range:
                # 创建随机森林模型
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测测试集
                y_pred = model.predict(X_test)
                
                # 计算 MAPE，确保真实值 y_test 非零
                non_zero_mask = y_test != 0
                y_true_non_zero = y_test[non_zero_mask]
                y_pred_non_zero = y_pred[non_zero_mask]
                mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
                
                # 打印参数组合和对应的 MAPE
                print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, "
                      f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
                print(f"MAPE: {mape:.2f}%\n")
                
                # 将 MAPE 上传到 wandb
                wandb.log({"MAPE": mape, "n_estimators": n_estimators, 
                           "max_depth": max_depth, "min_samples_split": min_samples_split,
                           "min_samples_leaf": min_samples_leaf})

# 完成 wandb 实验
 

# 打印 MAPE 值
#print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 可视化实际值与预测值的对比
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
plt.plot(y_pred, label="Predicted", linestyle='-', marker='x')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Confirmed Cases')
plt.title('Actual vs Predicted Confirmed Cases')
plt.savefig('plot_xgboost.png')
plt.show()
wandb.log({"chart": wandb.Image('plot_xgboost.png')})

# 完成 wandb 实验
wandb.finish()
