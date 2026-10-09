"""pipeline/evaluate.py — reporting helpers for the (archived) sentiment Random Forest."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.data import FIGURES


def plot_predictions(y_test, y_pred_log, asset="BTC", n=100):
    actual = np.exp(np.asarray(y_test)[:n])
    predicted = np.exp(np.asarray(y_pred_log)[:n])
    plt.figure(figsize=(12, 4))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted (RF)")
    plt.title(f"{asset} — Sentiment-Aware Volatility Forecast")
    plt.legend()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"legacy_sentiment_rf_{asset.lower()}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def feature_importance(model, features):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature importance (impurity, training data):")
    print(imp)
    return imp
