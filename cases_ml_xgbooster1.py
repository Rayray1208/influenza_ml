import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import wandb
import matplotlib.pyplot as plt

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
    wandb.init(project="flu_cases_prediction", name="xgboost_model")

    data = pd.read_csv("data.csv")
    data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
    data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

    X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases']]
    y = data['ConfirmedCases']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    param_grid = {
    'n_estimators': [500],
    'max_depth': [20],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0.1],
    'min_child_weight': [3]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(mape_scorer, greater_is_better=False),
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print(f"Best MAPE: {-grid_search.best_score_:.2f}%")
    
    for i, (params, mean_score, std_score) in enumerate(zip(grid_search.cv_results_['params'],
                                                            grid_search.cv_results_['mean_test_score'],
                                                            grid_search.cv_results_['std_test_score'])):
        wandb.log({
            "Fold": i,
            "Mean MAPE": -mean_score,
            "Std MAPE": std_score,
            "Parameters": params
        })

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)

    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    wandb.log({"Final MAPE": mape, "Final MSE": mse, "Final MAE": mae})

    plt.figure(figsize=(10,6))
    plt.plot(y_test.values, label="Actual", linestyle='--', marker='o')
    plt.plot(y_pred, label="Predicted", linestyle='--', marker='x')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Confirmed Cases')
    plt.title('Actual vs Predicted Confirmed Cases')
    plt.savefig('plot_xgboost.png')
    plt.close()

    wandb.log({"chart": wandb.Image('plot_xgboost.png')})

if __name__ == "__main__":
    main()