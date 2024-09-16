1. 隨機森林 mape :2=125.92%
2. 隨機森林 mape(
    Parameters: n_estimators=500, max_depth=20, min_samples_split=2, min_samples_leaf=1
MAPE: 123.08%
) : 123.08%
3. 支持向量機 SVR mape :140.77%
4. bgboost 回歸 mape:
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
    2.最接近 40% 的参数组合是: {
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
