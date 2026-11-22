import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


def evaluate_baseline(y_true: pd.Series) -> dict:
    """Simple baseline predicting the historical mean revenue."""
    baseline_pred = np.full(shape=len(y_true), fill_value=y_true.mean())
    rmse = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))
    mae = float(mean_absolute_error(y_true, baseline_pred))
    return {"model": "baseline_mean", "rmse": rmse, "mae": mae}


def train_and_compare_models(X_train, y_train, X_test, y_test) -> dict:
    """Train multiple regression models and compare against baseline."""
    results = [evaluate_baseline(y_test)]

    candidates = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    best_model = None
    best_rmse = float("inf")
    best_name = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        results.append({"model": name, "rmse": rmse, "mae": mae})

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "comparison": results,
    }


def predict_revenue(model, features_df: pd.DataFrame) -> np.ndarray:
    """Generate revenue predictions from feature DataFrame."""
    return model.predict(features_df)
