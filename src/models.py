"""Model training and evaluation utilities for the NBA Statistics project."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Win Prediction (Classification)
# ---------------------------------------------------------------------------

def build_win_prediction_dataset(games: pd.DataFrame) -> tuple:
    """Build feature matrix X and label vector y for win prediction.

    Uses rolling home/away stats that must already be present in `games`
    (added by ``features.add_rolling_team_stats``).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series (1 = home team wins, 0 = visitor wins)
    """
    feature_cols = [
        "HOME_ROLLING_PTS",
        "HOME_ROLLING_PTS_ALLOWED",
        "AWAY_ROLLING_PTS",
        "AWAY_ROLLING_PTS_ALLOWED",
    ]
    missing = [c for c in feature_cols if c not in games.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            "Run features.add_rolling_team_stats() first."
        )
    df = games[feature_cols + ["HOME_TEAM_WINS"]].dropna()
    X = df[feature_cols]
    y = df["HOME_TEAM_WINS"]
    return X, y


def train_logistic_regression(X_train, y_train) -> Pipeline:
    """Train a Logistic Regression win-prediction model."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest win-prediction model."""
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Evaluate a classifier and return a dict of metrics."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    return metrics


# ---------------------------------------------------------------------------
# Player Scoring Prediction (Regression)
# ---------------------------------------------------------------------------

def build_scoring_dataset(player_season: pd.DataFrame) -> tuple:
    """Build feature matrix X and target vector y for per-game scoring prediction.

    Parameters
    ----------
    player_season:
        Output of ``features.aggregate_player_season_stats()``.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series (PTS_PG — points per game)
    """
    feature_cols = ["MIN_PG", "AST_PG", "REB_PG", "STL_PG", "BLK_PG"]
    missing = [c for c in feature_cols if c not in player_season.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = player_season[feature_cols + ["PTS_PG"]].dropna()
    X = df[feature_cols]
    y = df["PTS_PG"]
    return X, y


def train_ridge_regression(X_train, y_train, alpha: float = 1.0) -> Pipeline:
    """Train a Ridge Regression player-scoring model."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=alpha)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_regressor(model, X_test, y_test) -> dict:
    """Evaluate a regression model and return a dict of metrics."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {
        "rmse": rmse,
        "r2": r2_score(y_test, y_pred),
    }


# ---------------------------------------------------------------------------
# Convenience: full train/test split + cross-validation report
# ---------------------------------------------------------------------------

def run_cv_report(model, X, y, cv: int = 5, scoring: str = "accuracy") -> dict:
    """Run k-fold cross-validation and return mean ± std scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "cv_scores": scores.tolist(),
    }
