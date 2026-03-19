# March Madness 2026 — ML Bracket Prediction Model

A machine learning pipeline that predicts NCAA Tournament outcomes using historical game data from the [BallDontLie NCAAB API](https://ncaab.balldontlie.io). The model predicts point spreads for individual matchups, converts them to win probabilities, and simulates the full bracket 10,000 times via Monte Carlo to generate round-by-round advancement probabilities for every team in the field.

**[View Live Predictions →](https://anthonygheen.github.io/MarchMadness2026/)**

---

## How It Works

### 1. Data Collection (`collect_data.py`)
Pulls historical NCAAB data from the BallDontLie API (GOAT tier required) across multiple seasons:
- **Games** — completed game results with scores (target variable source)
- **Team season stats** — per-team shooting percentages, win rates, records
- **Standings** — conference records, home/away splits, playoff seeds
- **Rankings** — AP Poll and Coaches Poll weekly rankings
- **Bracket** — current tournament field, seeds, and matchups (paginated)
- **Betting odds** — market lines for calibration reference

All data is saved as Parquet files in `data/`.

### 2. Feature Engineering
Features are built at the **team-season level** using only pre-game knowable information — no in-game stats that would constitute data leakage. Every matchup produces three feature variants per stat: `home_<feature>`, `away_<feature>`, and `diff_<feature>` (the differential).

**Team features:**
- Shooting efficiency: FG%, 3P%, FT%
- Season record: overall win%, conference win%, home/away/conference W-L
- Poll rankings: AP rank, Coaches rank, ranked/unranked binary
- Tournament seeding: conference playoff seed

**Conference-level features** (computed from game data, no extra API calls):
- `conf_pace` — average total points per game in conference games (tempo proxy)
- `conf_strength` — win% of conference teams in non-conference games only
- `conf_depth` — average margin of victory in conference games (competitiveness)

### 3. Model Training (`train_model.py`)

**Target variable:** point differential (spread), not binary win/loss. Predicting spread gives richer gradient signal and converts naturally to win probability via sigmoid: `P(win) = sigmoid(spread / 10)`.

**Tournament game weighting:** Games played in mid-March through early April are weighted 3x during training (date-based detection, not keyword matching which proved unreliable in the BallDontLie API).

**Validation:** Time-series cross-validation — trains on seasons 0..k, tests on season k+1. No random splits, which would leak future data into training. `PredefinedSplit` is used to enforce temporal structure during grid search as well.

**Models evaluated with grid search:**
| Model | Tuned Parameters |
|-------|-----------------|
| Linear Regression | Baseline (no tuning) |
| Ridge | alpha |
| Lasso | alpha |
| ElasticNet | alpha, l1_ratio |
| LinearSVR | C, epsilon |
| Gradient Boosting | n_estimators, max_depth, learning_rate, subsample, min_samples_leaf |

**Selection metric:** log loss on win probability predictions (penalizes confident wrong picks).

### 4. Bracket Pull (`pull_bracket.py`)
Pulls and validates the current tournament bracket from the API before simulating. The bracket endpoint paginates — without full pagination only the first region's games are returned. Validates:
- 32 Round of 64 games present (4 complete regions of 8)
- All seeds 1-16 represented
- TBD slot count matches active play-in games

### 5. Bracket Simulation (`predict_bracket.py`)

**Play-in handling:** Teams already confirmed in round-1 are excluded from play-in simulation to prevent double-counting. Play-in winners are assigned to TBD slots by bracket_location order (deterministic 1-to-1 mapping).

**Region simulation (per region, per simulation):**
1. Build 16-team R64 field, substituting play-in winners for TBD slots
2. Simulate each R64 game individually → 8 winners enter R32
3. Pair R64 winners sequentially for R32 (game 1 winner vs game 2 winner, etc.) — correctly implements standard bracket structure
4. Simulate R32 → 4 teams (S16)
5. Simulate S16 → 2 teams (E8)
6. Simulate E8 regional final → 1 regional champion

**Neutral court correction:** Each matchup predicted twice (A as home, B as home) and probabilities averaged to remove the home court advantage baked into a model trained on regular season games.

**Win probabilities** for all ~2,100 possible matchups are precomputed and cached before the simulation loop — only dictionary lookups inside the loop.

**Monte Carlo:** Simulates 10,000 full brackets. Each simulation draws random outcomes weighted by model probabilities. Outputs round-by-round advancement probability for every team.

**Sanity checks:** Champion probs sum to ~1.0, Final Four probs sum to ~4.0, R64 probs sum to ~64.0.

---

## Results

Results are published at the GitHub Pages link above with two views:

- **Bracket view** — ESPN-style bracket with top/bottom region split, seed-colored team cards, probability heat bars, and projected champion callout
- **Probability table** — sortable and filterable with heatmap coloring across all rounds

---

## Model Performance

Evaluated on held-out seasons using time-series cross-validation:

| Metric | Value |
|--------|-------|
| ROC AUC | ~0.82 |
| Brier Score | ~0.163 |
| Accuracy | ~76.5% |
| MAE (spread) | ~9.5 pts |

A Brier score of 0.163 compares favorably to the no-skill baseline of 0.25 (always predicting 50/50). Market-implied probabilities from closing lines typically achieve ~0.18.

---

## Project Structure

```
MarchMadness2026/
│
├── collect_data.py          # Pulls all historical data from BallDontLie API
├── train_model.py           # Feature engineering, grid search, model selection
├── pull_bracket.py          # Pulls and validates current tournament bracket
├── predict_bracket.py       # Monte Carlo bracket simulation and report
├── debug_bracket.py         # Bracket structure debugging utility
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
│   ├── grid_search_results.csv
│   ├── evaluation_plots.png
│   ├── bracket_raw.parquet
│   ├── bracket_predictions.csv
│   └── bracket_report.txt
│
├── docs/                    # GitHub Pages site
│   ├── index.html           # Bracket visualization (ESPN-style, light mode)
│   └── bracket_predictions.csv
│
├── .env                     # API key (gitignored)
├── .gitignore
├── activate.ps1             # Windows PowerShell venv activation script
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

# Create and activate virtual environment (Windows)
python -m venv venv
.\activate.ps1

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

### Step 1 — Pull historical training data
```bash
python collect_data.py --seasons 2018 2019 2020 2021 2022 2023 2024 2025 --skip-player-stats
```

### Step 2 — Train and evaluate models (with grid search)
```bash
python train_model.py --target-season 2024 --cv-splits 5 --tournament-weight 3.0
```

Grid search runs on all cores by default (`--n-jobs -1`). Expect ~30 minutes for the full grid.

### Step 3 — Pull and validate bracket
```bash
python pull_bracket.py --season 2025
```

Wait for `BRACKET STATUS: COMPLETE` before proceeding.

### Step 4 — Generate bracket predictions
```bash
python predict_bracket.py --season 2025 --simulations 10000
```

Check the sanity output — Champion probs should sum to ~1.0, Final Four to ~4.0.

### Step 5 — Deploy to GitHub Pages
```bash
copy models\bracket_predictions.csv docs\bracket_predictions.csv
git add docs\bracket_predictions.csv
git commit -m "Update predictions"
git push
```

GitHub Pages refreshes within ~2 minutes. Hard refresh with `Ctrl+Shift+R` if the old version is cached.

---

## Key Design Decisions

**Why predict spread instead of winner?**
Regression on point differential provides richer training signal — a 1-point win and a 20-point win are treated differently, which better reflects team quality. The spread converts to win probability via sigmoid, identical to how sportsbooks derive implied probabilities from lines.

**Why time-series CV instead of random k-fold?**
Random splits allow future seasons into training folds, artificially inflating CV metrics. Time-series CV enforces temporal structure: you can only use information available before the game you're predicting. The same structure is enforced during grid search using `PredefinedSplit`.

**Why Monte Carlo instead of deterministic bracket?**
A deterministic bracket always picks the higher-probability team, compounding errors across 6 rounds and ignoring inherent tournament variance. Monte Carlo respects that a 30% underdog wins ~30% of the time and produces calibrated round-by-round probabilities across all plausible bracket outcomes.

**Why average home/away orientations for neutral court games?**
The model was trained on regular season games with real home teams, implicitly encoding home court advantage. Tournament games are played at neutral sites, so predicting each matchup in both orientations and averaging removes that bias.

**Why date-based tournament game detection?**
The BallDontLie API `period_detail` field is unreliable for identifying tournament games — it only says "Final" for completed games regardless of context. Date-based detection (mid-March through early April) correctly identifies all NCAA Tournament games.

**Why bracket_location-based TBD mapping?**
Play-in winners are assigned to TBD slots using bracket_location order rather than seed matching. Seed matching breaks when two play-in games share the same seed (e.g. two 11-seed games). Location-based mapping creates a deterministic 1-to-1 assignment that mirrors the physical bracket structure.

---

## Limitations

- No direct measure of player quality, injuries, or roster turnover — all features are aggregate season statistics
- Tournament sample sizes are small (~67 games/year), making pure tournament-only validation noisy
- Conference features computed from same-season game data create mild circularity for early-season predictions
- The BallDontLie bracket endpoint may be incomplete early in Selection Week — run `pull_bracket.py` and wait for `BRACKET STATUS: COMPLETE` before simulating

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data collection | `requests`, `pandas` |
| Feature storage | `pyarrow` (Parquet) |
| Modeling | `scikit-learn` |
| Hyperparameter tuning | `GridSearchCV` + `PredefinedSplit` |
| Serialization | `joblib` |
| Visualization | `matplotlib` |
| Frontend | Vanilla HTML/CSS/JS |
| Fonts | Barlow Condensed (Google Fonts) |
| Hosting | GitHub Pages |

---

## License

MIT