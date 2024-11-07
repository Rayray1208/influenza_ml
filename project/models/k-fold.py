import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 自定义评分函数：MAPE
def mape_scorer(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# 创建示例数据集（使用make_regression来创建回归任务）
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# 创建分组（例如：每10个数据为一组，或者根据某个特征指定分组）
# 这里假设每个样本的组ID为样本索引（为示例使用）
groups = np.array([i % 10 for i in range(len(X))])  # 分成10组

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, test_size=0.2, random_state=42)

# XGBoost模型
xgb = XGBRegressor(random_state=42)
param_grid_xgb = {
        'n_estimators': [100, 200, 300, 500],  # 树的数量
        'max_depth': [3, 5, 7, 10, 15],  # 树的最大深度
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率
    }
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=GroupKFold(n_splits=5), scoring=make_scorer(mape_scorer), n_jobs=-1)
grid_search_xgb.fit(X_train, y_train, groups=groups_train)

# SVM模型
svm = SVR()
param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.1, 1],
        'epsilon': [0.1, 0.2, 0.5]
    }
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=GroupKFold(n_splits=5), scoring=make_scorer(mape_scorer), n_jobs=-1)
grid_search_svm.fit(X_train, y_train, groups=groups_train)

# 线性回归模型
lr = LinearRegression()
param_grid_lr = {}  # 线性回归没有需要调优的超参数
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=GroupKFold(n_splits=5), scoring=make_scorer(mape_scorer), n_jobs=-1)
grid_search_lr.fit(X_train, y_train, groups=groups_train)

# 输出每个模型的交叉验证结果
print("XGBoost Results")
print(grid_search_xgb.cv_results_['mean_test_score'], grid_search_xgb.cv_results_['std_test_score'])

print("SVM Results")
print(grid_search_svm.cv_results_['mean_test_score'], grid_search_svm.cv_results_['std_test_score'])

print("Linear Regression Results")
print(grid_search_lr.cv_results_['mean_test_score'], grid_search_lr.cv_results_['std_test_score'])

# 可视化三个模型的交叉验证结果
plt.figure(figsize=(12, 8))

# 绘制XGBoost的交叉验证得分
plt.plot(grid_search_xgb.cv_results_['mean_test_score'], label='XGBoost Mean Test Score')

# 绘制SVM的交叉验证得分
plt.plot(grid_search_svm.cv_results_['mean_test_score'], label='SVM Mean Test Score')


plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Mean Test Score (MAPE)')
plt.legend()
plt.title('Cross-Validation Results Comparison')
plt.show()

# 输出最佳模型和参数
print("Best XGBoost Model:", grid_search_xgb.best_params_)
print("Best SVM Model:", grid_search_svm.best_params_)
print("Best Linear Regression Model:", grid_search_lr.best_params_)