import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# 数据加载
data = pd.read_csv("merged_file_with_seasons.csv")
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

# 预测 ExcludedCases 的函数
def predict_new_value_ec(year_week):
    year = int(str(year_week)[:4])
    week = int(str(year_week)[4:])
    
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
    
    # 构建输入特征并预测
    input_data = pd.DataFrame({'Year': [year], 'Week': [week]})
    prediction = model.predict(input_data)
    
    return prediction[0]

# 预测 AverageTemperature 的函数
def predict_new_value_atp(year_week):
    year = int(str(year_week)[:4])
    week = int(str(year_week)[4:])
    
    # 特征和目标变量
    X = data[['Year', 'Week']]
    y = data['AverageTemperature']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用最佳参数创建模型
    best_params = {
        'reg_lambda': 0.1, 
        'reg_alpha': 1, 
        'n_estimators': 700, 
        'min_child_weight': 9, 
        'max_depth': 10, 
        'learning_rate': 0.2, 
        'gamma': 0.3
    }
    model = xgb.XGBRegressor(**best_params)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 构建输入特征并预测
    input_data = pd.DataFrame({'Year': [year], 'Week': [week]})
    prediction = model.predict(input_data)
    
    return prediction[0]

# 预测 ConfirmedCases 的主函数
def predict_new_value(year_week):
    year = int(str(year_week)[:4])
    week = int(str(year_week)[4:])
    
    # 预测 ExcludedCases 和 AverageTemperature
    predicted_value_excluded = predict_new_value_ec(year_week)
    predicted_value_averagetemperature = predict_new_value_atp(year_week)
    
    # 构建输入特征
    input_data = pd.DataFrame({
        'Year': [year],
        'Week': [week],
        'ExcludedCases': [predicted_value_excluded],  # 使用之前预测的值
        'PendingCases': [0],    # 或其他适当的默认值
        'AverageTemperature': [predicted_value_averagetemperature]  # 使用之前预测的值
    })
    
    # 使用最佳参数创建 ConfirmedCases 模型
    best_params = {
        'reg_lambda': 1, 
        'reg_alpha': 0,
        'n_estimators': 300, 
        'min_child_weight': 1, 
        'max_depth': 15, 
        'learning_rate': 0.05, 
        'gamma': 0.1
    }
    model = xgb.XGBRegressor(**best_params)
    
    # 训练 ConfirmedCases 模型
    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', 'AverageTemperature']]
    y = data['ConfirmedCases']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    # 进行预测
    prediction = model.predict(input_data)
    return prediction[0]

# 示例：输入要预测的 YearWeek 值
year_week_input = int(input("请输入要预测的 YearWeek 值（例如 202517）："))
predicted_value = predict_new_value(year_week_input)
print(f"预测的确诊病例数为: {predicted_value:.2f}")