"""Data loading and cleaning utilities for the NBA Statistics project.

Raw CSV files are expected in data/raw/ (downloaded from Kaggle):
  - games.csv
  - games_details.csv
  - players.csv
  - ranking.csv
  - teams.csv
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _path(filename: str) -> str:
    return os.path.join(RAW_DIR, filename)


def load_games(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load and clean the games dataset.

    Returns a DataFrame with one row per game containing:
        GAME_DATE_EST, GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID,
        PTS_home, PTS_away, HOME_TEAM_WINS, and season information.
    """
    path = os.path.join(raw_dir, "games.csv")
    df = pd.read_csv(path, parse_dates=["GAME_DATE_EST"])
    df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)
    # Drop rows where both team scores are missing
    df = df.dropna(subset=["PTS_home", "PTS_away"])
    df["PTS_home"] = df["PTS_home"].astype(int)
    df["PTS_away"] = df["PTS_away"].astype(int)
    df["HOME_TEAM_WINS"] = df["HOME_TEAM_WINS"].astype(int)
    return df


def load_games_details(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load and clean the per-player game details dataset.

    Returns a DataFrame with one row per player per game.
    """
    path = os.path.join(raw_dir, "games_details.csv")
    df = pd.read_csv(path)
    # Parse minutes column (format can be "MM:SS" or float)
    if "MIN" in df.columns:
        df["MIN"] = df["MIN"].apply(_parse_minutes)
    # Numeric stat columns
    stat_cols = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                 "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS"]
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def load_players(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load the players metadata dataset."""
    path = os.path.join(raw_dir, "players.csv")
    return pd.read_csv(path)


def load_ranking(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load the team ranking / standings dataset."""
    path = os.path.join(raw_dir, "ranking.csv")
    df = pd.read_csv(path, parse_dates=["STANDINGSDATE"])
    return df.sort_values("STANDINGSDATE").reset_index(drop=True)


def load_teams(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load the teams metadata dataset."""
    path = os.path.join(raw_dir, "teams.csv")
    return pd.read_csv(path)


def _parse_minutes(value) -> float:
    """Convert a 'MM:SS' string or numeric value to total minutes (float)."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0
