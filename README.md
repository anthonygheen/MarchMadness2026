# March Madness 2026 — ML Bracket Prediction Model

A machine learning pipeline that predicts NCAA Tournament outcomes using historical game data from the [BallDontLie NCAAB API](https://ncaab.balldontlie.io). The model predicts point spreads for individual matchups, converts them to win probabilities, and simulates the full bracket 10,000 times via Monte Carlo to generate round-by-round advancement probabilities for every team in the field.

**[View Live Predictions →](https://anthonygheen.github.io/MarchMadness2026/)**

---

## How It Works

### 1. Data Collection (`collect_data.py`)
Pulls historical NCAAB data from the BallDontLie API across multiple seasons:
- **Games** — completed game results with scores (target variable source)
- **Team season stats** — per-team shooting percentages, win rates, records
- **Standings** — conference records, home/away splits, playoff seeds
- **Rankings** — AP Poll and Coaches Poll weekly rankings
- **Bracket** — current tournament field, seeds, and matchups
- **Betting odds** — market lines for calibration reference

All data is saved as Parquet files in `data/`.

### 2. Feature Engineering
Features are built at the **team-season level** using only pre-game knowable information — no in-game stats that would constitute data leakage. Every matchup gets three feature variants: `home_<feature>`, `away_<feature>`, and `diff_<feature>` (the differential, which tends to be the most predictive).

**Team features used:**
- Shooting efficiency: FG%, 3P%, FT%
- Season record: overall win%, conference win%, home/away/conference W-L
- Poll rankings: AP rank, Coaches rank, ranked/unranked binary
- Tournament seeding: conference playoff seed

**Conference-level features** (computed from game data, no extra API calls):
- `conf_pace` — average total points per game in conference games (tempo proxy)
- `conf_strength` — win% of conference teams in non-conference games only
- `conf_depth` — average margin of victory in conference games (competitiveness)

### 3. Model Training (`train_model.py`)
- **Target variable:** point differential (spread), not binary win/loss
- Predicting spread gives richer gradient signal and naturally converts to win probability via a sigmoid function: `P(win) = sigmoid(spread / 10)`
- **Validation:** time-series cross-validation — trains on seasons 0..k, tests on season k+1. No random splits, which would leak future data into training
- **Tournament games** are upweighted 3x during training since they represent the inference context
- **Models evaluated:** Linear Regression, Ridge, Lasso, ElasticNet, Gradient Boosting
- **Selection metric:** log loss on win probability predictions (penalizes confident wrong picks)

The best model and feature importance rankings are saved to `models/`.

### 4. Bracket Simulation (`predict_bracket.py`)
- Pulls the live tournament bracket from the API
- Joins current-season team features onto each bracket entry
- **Neutral court correction:** each matchup is predicted twice (A as home, B as home) and the probabilities are averaged to remove home court bias
- Win probabilities for all ~1,100 possible matchups are precomputed and cached before simulation begins
- **Monte Carlo:** simulates the full bracket 10,000 times, sampling game outcomes from model probabilities each iteration
- Outputs round-by-round advancement probability for every team

---

## Results

Results are published at the GitHub Pages link above. The visualization includes:
- **Bracket view** — all teams organized by region with per-round probabilities and color-coded championship odds
- **Probability table** — fully sortable and filterable table with heatmap coloring

---

## Model Performance

Evaluated on held-out seasons using time-series cross-validation:

| Metric | Value |
|--------|-------|
| ROC AUC | ~0.82 |
| Brier Score | ~0.163 |
| Accuracy | ~76.5% |
| MAE (spread) | ~9.5 pts |

A Brier score of 0.163 compares favorably to the no-skill baseline of 0.25 (always predicting 50/50) and is in the range of well-calibrated college basketball models. Market-implied probabilities from closing lines typically achieve ~0.18.

---

## Project Structure

```
MarchMadness2026/
│
├── collect_data.py          # Pulls all historical data from BallDontLie API
├── train_model.py           # Feature engineering, model selection, evaluation
├── predict_bracket.py       # Bracket pull, Monte Carlo simulation, report
│
├── data/                    # Parquet data files (gitignored)
│   ├── games.parquet
│   ├── team_features.parquet
│   ├── team_season_stats.parquet
│   ├── standings.parquet
│   ├── rankings.parquet
│   └── game_dataset.parquet
│
├── models/                  # Model artifacts (gitignored)
│   ├── best_model.joblib
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── evaluation_plots.png
│   ├── bracket_raw.parquet
│   ├── bracket_predictions.csv
│   └── bracket_report.txt
│
├── docs/                    # GitHub Pages site
│   ├── index.html
│   └── bracket_predictions.csv
│
├── .env                     # API key (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.11+
- [BallDontLie GOAT tier API key](https://app.balldontlie.io) ($39.99/mo) — required for team stats, bracket, and odds endpoints

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/MarchMadness2026.git
cd MarchMadness2026

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
BDL_API_KEY=your_api_key_here
```

---

## Usage

### Step 1 — Pull historical data
```bash
python collect_data.py --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

For current tournament predictions only, add `--skip-player-stats` to speed things up:
```bash
python collect_data.py --seasons 2025 --skip-player-stats
```

### Step 2 — Train and evaluate models
```bash
python train_model.py --target-season 2024 --cv-splits 5
```

Outputs model comparison table, feature importance, and evaluation plots to `models/`.

### Step 3 — Generate bracket predictions
```bash
python predict_bracket.py --season 2025 --simulations 10000
```

Outputs `bracket_predictions.csv` and `bracket_report.txt` to `models/`.

### Step 4 — Deploy to GitHub Pages
```bash
cp models/bracket_predictions.csv docs/
git add docs/bracket_predictions.csv
git commit -m "Update predictions"
git push
```

---

## Key Design Decisions

**Why predict spread instead of winner?**
Regression on point differential provides richer training signal — a 1-point win and a 20-point win are treated differently, which better reflects team quality. The spread is then converted to a win probability via sigmoid, identical to how sportsbooks derive implied probabilities from lines.

**Why time-series CV instead of random k-fold?**
Random splits would allow future seasons to appear in training folds, artificially inflating CV metrics. Time-series CV enforces the temporal structure of the problem: you can only use information available before the game you're predicting.

**Why Monte Carlo instead of a single deterministic bracket?**
A deterministic bracket always picks the higher-probability team to advance, which compounds errors across 6 rounds and ignores the inherent variance of tournament basketball. Monte Carlo simulations respect that upsets happen — a 30% underdog wins 3,000 times out of 10,000 — and produce calibrated round-by-round probabilities rather than a single brittle prediction.

**Why average home/away orientations for neutral court games?**
The model was trained on regular season games with real home teams, so it has home court advantage implicitly encoded. Tournament games are played at neutral sites, so predicting each matchup in both orientations and averaging removes that bias.

---

## Limitations

- Conference features are derived from the same season's game data, which creates a mild circularity for early-season predictions
- The model has no direct measure of player quality, injuries, or roster turnover — all team features are aggregate season statistics
- Tournament sample sizes are small (~67 games/year), making pure tournament-only validation noisy
- The BallDontLie API bracket may be incomplete early in Selection Week — predictions improve once all 64 teams and matchups are confirmed

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data collection | `requests`, `pandas` |
| Feature storage | `pyarrow` (Parquet) |
| Modeling | `scikit-learn` |
| Serialization | `joblib` |
| Visualization | `matplotlib` |
| Frontend | Vanilla HTML/CSS/JS |
| Hosting | GitHub Pages |

---

## License

MIT