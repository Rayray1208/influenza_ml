import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# 初始化 Weights & Biases


# 数据加载
data = pd.read_csv("merged_file_with_seasons.csv")
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)


# 特征和目标变量
X = data[['Year', 'Week']]
y = data['ExcludedCases']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用最佳参数创建模型
best_params = {
        'reg_lambda': 0, 
        'reg_alpha': 1, 
        'n_estimators': 700, 
        'min_child_weight': 1, 
        'max_depth': 5, 
        'learning_rate': 0.2, 
        'gamma': 0.3
    }

model = xgb.XGBRegressor(**best_params)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算评估指
def predict_new_value(year_week):
    year = int(str(year_week)[:4])
    week = int(str(year_week)[4:])
    
    # 构建输入特征
    input_data = pd.DataFrame({
        'Year': [year],
        'Week': [week]
    })
    
    # 进行预测
    prediction = model.predict(input_data)
    return prediction[0]

# 示例：输入要预测的值
year_week_input = int(input("请输入要预测的 YearWeek 值（例如 202517）："))
predicted_value_excluded = predict_new_value(year_week_input)
print(f"EXCLUDED的病例数为: {predicted_value_excluded:.2f}")