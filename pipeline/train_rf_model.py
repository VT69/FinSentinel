"""pipeline/train_rf_model.py — fit/score/save the (archived) sentiment Random Forest."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipeline.data import MODELS

RF_PARAMS = dict(n_estimators=600, max_depth=8, min_samples_leaf=10, max_features="sqrt",
                 random_state=42, n_jobs=-1)


class TooLittleDataError(RuntimeError):
    pass


def train_random_forest(X_train, X_test, y_train, y_test, asset="BTC"):
    # With fewer than 2 * min_samples_leaf rows no tree can split: every tree is a single
    # leaf and the "forest" is a constant (this is what produced the old models/rf_btc.pkl).
    min_rows = 2 * RF_PARAMS["min_samples_leaf"]
    if len(X_train) < min_rows:
        raise TooLittleDataError(
            f"{len(X_train)} training rows < {min_rows} (= 2 * min_samples_leaf): the forest would be a "
            "constant predictor. Collect more sentiment history before training this model.")

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    pred_vol, actual_vol = np.exp(preds), np.exp(y_test)
    rmse = np.sqrt(mean_squared_error(actual_vol, pred_vol))
    mae = mean_absolute_error(actual_vol, pred_vol)
    print(f"\n{asset} Random Forest — RMSE {rmse:.6f}  MAE {mae:.6f}  (n_test={len(y_test)})")

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"legacy_sentiment_rf_{asset.lower()}.pkl")
    return model, rmse, mae
