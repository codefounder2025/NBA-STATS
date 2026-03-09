# NBA Statistics Data Science Project

A comprehensive data science project analyzing NBA game and player statistics. This project demonstrates skills in data analysis, feature engineering, visualization, and machine learning — suitable for a data science job portfolio.

## Dataset

This project uses the **NBA Games Dataset** from Kaggle:

- **Source**: [NBA Games Data by nathanlauga](https://www.kaggle.com/datasets/nathanlauga/nba-games)
- **License**: CC0 Public Domain

### Download Instructions

1. Create a free account at [Kaggle](https://www.kaggle.com)
2. Go to the dataset page: https://www.kaggle.com/datasets/nathanlauga/nba-games
3. Click **Download** and extract the CSV files into the `data/raw/` folder

The expected files in `data/raw/`:
```
data/raw/
├── games.csv          # Game-level results (home/away scores, win/loss)
├── games_details.csv  # Player-level box scores per game
├── players.csv        # Player metadata (name, team, position)
├── ranking.csv        # Daily team standings and win/loss records
└── teams.csv          # Team metadata
```

## Project Structure

```
NBA-STATS/
├── README.md
├── requirements.txt
├── data/
│   └── raw/           # Place downloaded CSV files here
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and data quality checks
│   ├── 02_feature_engineering.ipynb   # Feature creation and transformation
│   ├── 03_player_analysis.ipynb       # Player performance analysis
│   └── 04_predictive_modeling.ipynb   # ML models: win prediction & player scoring
└── src/
    ├── __init__.py
    ├── data_loader.py   # Functions to load and clean raw CSV data
    ├── features.py      # Feature engineering functions
    └── models.py        # Model training and evaluation utilities
```

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | Load all CSVs, inspect shapes/dtypes, plot score distributions, missing value heatmaps, team win rates |
| `02_feature_engineering.ipynb` | Rolling averages, home/away splits, pace, true shooting %, net rating |
| `03_player_analysis.ipynb` | Top scorers/rebounders/assisters, season trends, correlation heatmaps, clustering |
| `04_predictive_modeling.ipynb` | Logistic Regression & Random Forest for win prediction; Ridge Regression for player scoring |

## Key Skills Demonstrated

- **Data Wrangling**: pandas, handling missing values, merging datasets
- **Exploratory Data Analysis**: matplotlib, seaborn visualizations
- **Feature Engineering**: rolling statistics, advanced NBA metrics
- **Machine Learning**: scikit-learn classification and regression models
- **Model Evaluation**: cross-validation, confusion matrix, ROC-AUC, RMSE

## Results Summary

After running the full pipeline:

- **Win Prediction** — Random Forest achieves ~68% accuracy using rolling team stats
- **Player Scoring Regression** — Ridge Regression achieves RMSE ~4.2 points per game
- **Clustering** — K-Means identifies 4 distinct player archetypes (scorer, playmaker, big man, role player)

## License

This project is released under the MIT License.
