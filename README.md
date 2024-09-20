1. 隨機森林 mape :2=125.92%
2. 隨機森林 mape(
    Parameters: n_estimators=500, max_depth=20, min_samples_split=2, min_samples_leaf=1
MAPE: 123.08%
) : 123.08%
3. 支持向量機 SVR mape :140.77%
4. bgboost 回歸 mape:
        1.（純病例數數據）
        最接近 70% 的参数组合是: {
        'colsample_bytree': 1.0,
        'gamma': 0.1, 
        'learning_rate': 0.05, 
        'max_depth': 5, 
        'min_child_weight': 1, 
        'n_estimators': 1000, 
        'subsample': 0.8}, 对应的 MAPE 是: -70.64%
        最佳参数组合: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 0.8}
        最佳 MAPE: -70.64%
        最终模型 MAPE: 103.99%
        均方误差 (MSE): 224.34
        平均绝对误差 (MAE): 8.85
        2.引用合成天氣數據（模擬數據）
        最接近 40% 的参数组合是: {
        'colsample_bytree': 0.8, 
        'gamma': 0, 
        'learning_rate': 0.2, 
        'max_depth': 10, 
        'min_child_weight': 3, 
        'n_estimators': 100, 
        'subsample': 1.0
        }, 
        对应的 MAPE 是: -66.61%
        最佳参数组合: {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 7, 'n_estimators': 100, 'subsample': 0.6}
        最佳 MAPE: -218.92%
        最终模型 MAPE: 360.36%
        均方误差 (MSE): 1048.20
        平均绝对误差 (MAE): 18.64
        3.純數據
        param_grid = {
            'n_estimators': [500],
            'max_depth': [ 20],
            'learning_rate': [ 0.2],
            'subsample': [1.0],
            'colsample_bytree': [1.0],
            'gamma': [0.5],
            'min_child_weight': [1]
        }
        Best Parameters: {'colsample_bytree': 1.0, 'gamma': 0.5, 'learning_rate': 0.2, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 500, 'subsample': 1.0}
        Best MAPE: -85.62%
        Mean Absolute Percentage Error (MAPE): 101.13%
        Mean Squared Error (MSE): 369.24
        Mean Absolute Error (MAE): 10.14
        4.引用含有波動性的天氣數據
        最接近 40% 的参数组合是: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 7, 'n_estimators': 300, 'subsample': 0.8}, 对应的 MAPE 是: -75.42%
        最佳参数组合: {'colsample_bytree': 0.6, 'gamma': 0.3, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 7, 'n_estimators': 100, 'subsample': 0.6}
        最佳 MAPE: -216.46%
        最终模型 MAPE: 294.69%
        均方误差 (MSE): 506.07
        平均绝对误差 (MAE): 15.62
        1 Fitting 5 folds for each of 1 candidates, totalling 5 fits
 2 最接近 70% 的参数组合是: {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 3, 'n_estimators': 500, 'subsample': 0.8}, 对应的 MAPE 是: -82.54%
 3 最佳参数组合: {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 3, 'n_estimators': 500, 'subsample': 0.8}
 4 最佳 MAPE: -82.54%
 5 最终模型 MAPE: 99.01%
 6 均方误差 (MSE): 347.64
 7 平均绝对误差 (MAE): 10.19