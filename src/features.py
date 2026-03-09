"""Feature engineering utilities for the NBA Statistics project."""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Team-level rolling features
# ---------------------------------------------------------------------------

def add_rolling_team_stats(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling-average offensive / defensive stats to the games DataFrame.

    For each game, we compute the home team's and visitor team's rolling
    average points scored and allowed over the previous `window` games
    (excluding the current game to avoid data leakage).

    Parameters
    ----------
    games:
        Output of ``data_loader.load_games()``.
    window:
        Number of past games to average over.

    Returns
    -------
    DataFrame with additional columns:
        HOME_ROLLING_PTS, HOME_ROLLING_PTS_ALLOWED,
        AWAY_ROLLING_PTS, AWAY_ROLLING_PTS_ALLOWED
    """
    df = games.copy().sort_values("GAME_DATE_EST").reset_index(drop=True)

    # Build a combined record of every team's scoring history
    home = df[["GAME_DATE_EST", "HOME_TEAM_ID", "PTS_home", "PTS_away"]].copy()
    home.columns = ["date", "team_id", "pts_scored", "pts_allowed"]

    away = df[["GAME_DATE_EST", "VISITOR_TEAM_ID", "PTS_away", "PTS_home"]].copy()
    away.columns = ["date", "team_id", "pts_scored", "pts_allowed"]

    all_games = pd.concat([home, away], ignore_index=True).sort_values("date")

    # Compute per-team expanding / rolling means (shift(1) to exclude current game)
    rolling_scored = (
        all_games.groupby("team_id")["pts_scored"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    rolling_allowed = (
        all_games.groupby("team_id")["pts_allowed"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    all_games["rolling_scored"] = rolling_scored
    all_games["rolling_allowed"] = rolling_allowed

    # Re-split home/away
    n = len(df)
    home_roll = all_games.iloc[:n].reset_index(drop=True)
    away_roll = all_games.iloc[n:].reset_index(drop=True)

    df["HOME_ROLLING_PTS"] = home_roll["rolling_scored"]
    df["HOME_ROLLING_PTS_ALLOWED"] = home_roll["rolling_allowed"]
    df["AWAY_ROLLING_PTS"] = away_roll["rolling_scored"]
    df["AWAY_ROLLING_PTS_ALLOWED"] = away_roll["rolling_allowed"]

    return df


def add_point_differential(games: pd.DataFrame) -> pd.DataFrame:
    """Add a POINT_DIFF column (home - away) to the games DataFrame."""
    df = games.copy()
    df["POINT_DIFF"] = df["PTS_home"] - df["PTS_away"]
    return df


# ---------------------------------------------------------------------------
# Player-level advanced metrics
# ---------------------------------------------------------------------------

def add_true_shooting(details: pd.DataFrame) -> pd.DataFrame:
    """Add True Shooting Percentage (TS%) to the game details DataFrame.

    TS% = PTS / (2 * (FGA + 0.44 * FTA))
    """
    df = details.copy()
    denominator = 2 * (df["FGA"] + 0.44 * df["FTA"])
    df["TS_PCT"] = np.where(denominator > 0, df["PTS"] / denominator, 0.0)
    return df


def add_usage_rate(details: pd.DataFrame) -> pd.DataFrame:
    """Add a simplified Usage Rate proxy to the game details DataFrame.

    Usage ≈ (FGA + 0.44 * FTA + TO) / MIN  (per-minute possessions used)
    """
    df = details.copy()
    df["USAGE_RATE"] = np.where(
        df["MIN"] > 0,
        (df["FGA"] + 0.44 * df["FTA"] + df["TO"]) / df["MIN"],
        0.0,
    )
    return df


def aggregate_player_season_stats(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate game-level player stats to per-season totals and averages.

    Parameters
    ----------
    details:
        Output of ``data_loader.load_games_details()``, optionally enriched
        with advanced metrics.

    Returns
    -------
    DataFrame indexed by (PLAYER_ID, SEASON) with columns for totals and per-game
    averages of points, rebounds, assists, etc.
    """
    required = {"PLAYER_ID", "SEASON", "PTS", "REB", "AST", "STL", "BLK", "MIN"}
    if not required.issubset(details.columns):
        missing = required - set(details.columns)
        raise ValueError(f"Missing columns in details DataFrame: {missing}")

    grp = details.groupby(["PLAYER_ID", "SEASON"])
    agg = grp.agg(
        GP=("PTS", "count"),
        PTS_TOTAL=("PTS", "sum"),
        REB_TOTAL=("REB", "sum"),
        AST_TOTAL=("AST", "sum"),
        STL_TOTAL=("STL", "sum"),
        BLK_TOTAL=("BLK", "sum"),
        MIN_TOTAL=("MIN", "sum"),
    ).reset_index()

    for col in ["PTS", "REB", "AST", "STL", "BLK", "MIN"]:
        agg[f"{col}_PG"] = (agg[f"{col}_TOTAL"] / agg["GP"]).round(2)

    return agg
