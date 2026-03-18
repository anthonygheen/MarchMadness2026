"""
train_model.py
--------------
Model selection, hyperparameter tuning, and evaluation for March Madness
spread prediction.

Key features:
  - Tournament game upweighting using date-based detection (mid-March to
    early April) rather than period_detail keywords, which are unreliable
  - GridSearchCV with custom time-series folds — no data leakage from
    future seasons into hyperparameter search
  - Models tuned: Ridge, Lasso, ElasticNet, Gradient Boosting, LinearSVR
  - Linear Regression included as untuned baseline for comparison
  - Best model selected by CV log loss on win probability predictions

Outputs (models/ directory):
  - best_model.joblib
  - model_comparison.csv
  - feature_importance.csv
  - evaluation_plots.png
  - grid_search_results.csv    — full grid search scores for all models

Usage:
  python train_model.py [--data-dir data] [--model-dir models]
                        [--target-season 2024] [--cv-splits 5]
                        [--tournament-weight 3.0] [--n-jobs -1]
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit

import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    log_loss, brier_score_loss, roc_auc_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPREAD_SCALE = 10.0

# Tournament games run mid-March through first week of April
TOURNEY_MONTH_START = 3
TOURNEY_DAY_START   = 14
TOURNEY_MONTH_END   = 4
TOURNEY_DAY_END     = 7

# ---------------------------------------------------------------------------
# Feature allowlist
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "fg_pct", "fg3_pct", "ft_pct",
    "win_percentage", "conference_win_percentage",
    "home_wins", "home_losses", "away_wins", "away_losses",
    "conference_wins", "conference_losses", "wins", "losses",
    "ap_rank", "coach_rank", "is_ranked",
    "playoff_seed",
]

CONF_BASE_FEATURES = [
    "conf_pace",
    "conf_strength",
    "conf_depth",
]


def expand_allowlist(base_features: list[str]) -> list[str]:
    expanded = []
    for f in base_features:
        expanded.append(f"home_{f}")
        expanded.append(f"away_{f}")
        expanded.append(f"diff_{f}")
    return expanded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spread_to_prob(spread: np.ndarray, scale: float = SPREAD_SCALE) -> np.ndarray:
    return expit(spread / scale)


def is_tournament_game(df: pd.DataFrame) -> pd.Series:
    """
    Detects tournament games using game date rather than period_detail
    keywords, which are not reliably populated in the BallDontLie API.

    NCAA Tournament runs mid-March through first week of April each year.
    """
    if "date" not in df.columns:
        return pd.Series(False, index=df.index)

    dates = pd.to_datetime(df["date"], errors="coerce")
    month = dates.dt.month
    day   = dates.dt.day

    march_games = (month == TOURNEY_MONTH_START) & (day >= TOURNEY_DAY_START)
    april_games = (month == TOURNEY_MONTH_END)   & (day <= TOURNEY_DAY_END)

    return march_games | april_games


def regression_to_classification_metrics(
    y_true_spread: np.ndarray,
    y_pred_spread: np.ndarray,
    scale: float = SPREAD_SCALE,
) -> dict:
    y_prob        = spread_to_prob(y_pred_spread, scale)
    y_true_binary = (y_true_spread > 0).astype(int)

    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)

    return {
        "log_loss":    log_loss(y_true_binary, y_prob),
        "brier_score": brier_score_loss(y_true_binary, y_prob),
        "roc_auc":     roc_auc_score(y_true_binary, y_prob),
        "accuracy":    ((y_prob > 0.5) == y_true_binary).mean(),
    }


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fp = data_dir / "game_dataset.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"game_dataset.parquet not found in {data_dir}. "
            "Run collect_data.py first."
        )
    df        = pd.read_parquet(fp)
    tf        = pd.read_parquet(data_dir / "team_features.parquet") \
                if (data_dir / "team_features.parquet").exists() else pd.DataFrame()
    raw_games = pd.read_parquet(data_dir / "games.parquet") \
                if (data_dir / "games.parquet").exists() else pd.DataFrame()

    log.info("Loaded game dataset: %d rows, %d cols", *df.shape)
    return df, tf, raw_games


def build_conference_features(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    if "conference_id" not in team_features.columns:
        return team_features

    conf_map = (
        team_features[["team_id", "season", "conference_id"]]
        .dropna(subset=["conference_id"])
        .drop_duplicates()
    )

    g = games.copy()
    g = g.merge(
        conf_map.rename(columns={"team_id": "home_team_id", "conference_id": "home_conf_id"}),
        on=["home_team_id", "season"], how="left"
    )
    g = g.merge(
        conf_map.rename(columns={"team_id": "away_team_id", "conference_id": "away_conf_id"}),
        on=["away_team_id", "season"], how="left"
    )

    g["is_conf_game"] = g["home_conf_id"] == g["away_conf_id"]
    g["total_score"]  = g["home_score"] + g["away_score"]
    g["abs_margin"]   = g["point_diff"].abs()

    conf_games    = g[g["is_conf_game"]].copy()
    nonconf_games = g[~g["is_conf_game"]].copy()

    pace = (
        conf_games.groupby(["home_conf_id", "season"])["total_score"].mean()
        .reset_index()
        .rename(columns={"home_conf_id": "conference_id", "total_score": "conf_pace"})
    )
    depth = (
        conf_games.groupby(["home_conf_id", "season"])["abs_margin"].mean()
        .reset_index()
        .rename(columns={"home_conf_id": "conference_id", "abs_margin": "conf_depth"})
    )

    home_nc = nonconf_games[["home_conf_id", "season", "home_score", "away_score"]].copy()
    home_nc["win"] = (home_nc["home_score"] > home_nc["away_score"]).astype(int)
    home_nc = home_nc.rename(columns={"home_conf_id": "conference_id"})

    away_nc = nonconf_games[["away_conf_id", "season", "home_score", "away_score"]].copy()
    away_nc["win"] = (away_nc["away_score"] > away_nc["home_score"]).astype(int)
    away_nc = away_nc.rename(columns={"away_conf_id": "conference_id"})

    strength = (
        pd.concat([home_nc[["conference_id", "season", "win"]],
                   away_nc[["conference_id", "season", "win"]]])
        .groupby(["conference_id", "season"])["win"].mean()
        .reset_index()
        .rename(columns={"win": "conf_strength"})
    )

    conf_stats = (
        pace
        .merge(strength, on=["conference_id", "season"], how="outer")
        .merge(depth,    on=["conference_id", "season"], how="outer")
    )

    tf = team_features.merge(conf_stats, on=["conference_id", "season"], how="left")
    log.info(
        "Conference features: conf_pace %.1f-%.1f | "
        "conf_strength %.3f-%.3f | conf_depth %.1f-%.1f",
        tf["conf_pace"].min(),     tf["conf_pace"].max(),
        tf["conf_strength"].min(), tf["conf_strength"].max(),
        tf["conf_depth"].min(),    tf["conf_depth"].max(),
    )
    return tf


def prepare_features(
    df: pd.DataFrame,
    tf: pd.DataFrame,
    raw_games: pd.DataFrame,
    target_col: str = "point_diff",
    tournament_weight: float = 3.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()

    # --- Tournament game detection (date-based) ---
    tourney_flag = is_tournament_game(df)
    n_tourney    = tourney_flag.sum()
    log.info(
        "Tournament game detection: %d tournament games identified "
        "(weighted %.1fx) out of %d total",
        n_tourney, tournament_weight, len(df)
    )
    if n_tourney == 0:
        log.warning(
            "No tournament games detected. Check that 'date' column is "
            "present and games span mid-March through early April."
        )

    weights        = np.where(tourney_flag, tournament_weight, 1.0)
    sample_weights = pd.Series(weights, index=df.index)

    # --- Conference features ---
    if not tf.empty and not raw_games.empty:
        tf_with_conf = build_conference_features(raw_games, tf)

        conf_cols = ["team_id", "season"] + [
            f for f in CONF_BASE_FEATURES if f in tf_with_conf.columns
        ]

        home_conf = (
            tf_with_conf[conf_cols]
            .rename(columns={"team_id": "home_team_id"})
            .rename(columns={f: f"home_{f}" for f in CONF_BASE_FEATURES
                             if f in tf_with_conf.columns})
        )
        away_conf = (
            tf_with_conf[conf_cols]
            .rename(columns={"team_id": "away_team_id"})
            .rename(columns={f: f"away_{f}" for f in CONF_BASE_FEATURES
                             if f in tf_with_conf.columns})
        )

        df = df.merge(home_conf, on=["home_team_id", "season"], how="left")
        df = df.merge(away_conf, on=["away_team_id", "season"], how="left")

        for f in CONF_BASE_FEATURES:
            h, a = f"home_{f}", f"away_{f}"
            if h in df.columns and a in df.columns:
                df[f"diff_{f}"] = df[h] - df[a]
    else:
        log.warning("Skipping conference features — team_features or games not available.")

    # --- Apply allowlist ---
    all_base      = BASE_FEATURES + CONF_BASE_FEATURES
    allowed_cols  = expand_allowlist(all_base)
    available     = [c for c in allowed_cols if c in df.columns]
    missing       = [c for c in allowed_cols if c not in df.columns]

    if missing:
        log.warning("%d allowlist features not in dataset: %s", len(missing), missing)

    X = df[available].copy()

    # Drop columns with >60% missing
    high_missing = X.columns[X.isnull().mean() >= 0.6].tolist()
    if high_missing:
        log.warning("Dropping %d cols >60%% missing: %s", len(high_missing), high_missing)
        X = X.drop(columns=high_missing)

    y = df[target_col]

    log.info("Feature matrix: %d rows x %d cols", *X.shape)
    return X, y, sample_weights


# ---------------------------------------------------------------------------
# Model definitions and parameter grids
# ---------------------------------------------------------------------------

def build_base_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def get_model_configs() -> dict:
    """
    Returns model configurations as:
      name -> {
        pipeline:   sklearn Pipeline,
        param_grid: dict for GridSearchCV (None = no tuning),
      }

    Parameter grid keys use the pipeline step prefix:
      model__alpha, model__n_estimators, etc.
    """
    pre = build_base_preprocessor()

    return {
        "Linear Regression": {
            "pipeline": Pipeline([
                ("pre",   build_base_preprocessor()),
                ("model", LinearRegression()),
            ]),
            "param_grid": None,  # No hyperparameters to tune
        },

        "Ridge": {
            "pipeline": Pipeline([
                ("pre",   build_base_preprocessor()),
                ("model", Ridge()),
            ]),
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
            },
        },

        "Lasso": {
            "pipeline": Pipeline([
                ("pre",   build_base_preprocessor()),
                ("model", Lasso(max_iter=10000)),
            ]),
            "param_grid": {
                "model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            },
        },

        "ElasticNet": {
            "pipeline": Pipeline([
                ("pre",   build_base_preprocessor()),
                ("model", ElasticNet(max_iter=10000)),
            ]),
            "param_grid": {
                "model__alpha":    [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
                "model__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
            },
        },

        "LinearSVR": {
            "pipeline": Pipeline([
                ("pre",   build_base_preprocessor()),
                ("model", LinearSVR(max_iter=5000, dual=True)),
            ]),
            "param_grid": {
                "model__C":       [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
                "model__epsilon": [0.0, 0.1, 0.5, 1.0, 2.0],
            },
        },

        "Gradient Boosting": {
            "pipeline": Pipeline([
                ("pre",   build_base_preprocessor()),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]),
            "param_grid": {
                "model__n_estimators":  [100, 200, 300, 500],
                "model__max_depth":     [3, 4, 5],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__subsample":     [0.7, 0.8, 0.9],
                "model__min_samples_leaf": [10, 20, 30],
            },
        },
    }


# ---------------------------------------------------------------------------
# Time-series CV folds
# ---------------------------------------------------------------------------

def build_ts_folds(
    seasons: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Builds time-series cross-validation folds keyed on season.
    Each fold: train on seasons 0..k, test on season k+1.
    Returns list of (train_indices, test_indices).
    """
    unique_seasons = np.sort(np.unique(seasons))

    if len(unique_seasons) < n_splits + 1:
        n_splits = len(unique_seasons) - 1
        log.warning("Reduced CV splits to %d based on available seasons", n_splits)

    folds = []
    for i in range(n_splits):
        cutoff    = unique_seasons[-(n_splits - i)]
        train_idx = np.where(seasons < cutoff)[0]
        test_idx  = np.where(seasons == cutoff)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))

    return folds


def folds_to_predefined_split(
    n_samples: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> PredefinedSplit:
    """
    Converts fold indices to sklearn PredefinedSplit for use in GridSearchCV.

    PredefinedSplit assigns each sample a fold number (-1 = training only).
    We use the last fold's test set as the validation set for grid search,
    which is the most recent season — the best proxy for tournament performance.
    """
    test_fold = np.full(n_samples, -1, dtype=int)
    # Use only the last fold for grid search validation
    # (most recent season = closest to tournament context)
    _, test_idx = folds[-1]
    test_fold[test_idx] = 0
    return PredefinedSplit(test_fold)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    name: str,
    pipeline,
    param_grid: dict | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights_train: pd.Series,
    cv_split: PredefinedSplit,
    n_jobs: int,
) -> tuple:
    """
    Runs GridSearchCV for a single model. Returns (best_pipeline, results_df).

    For models that support sample_weight (Ridge, Lasso, ElasticNet, GB),
    weights are passed via fit_params. LinearSVR also supports sample_weight.
    Linear Regression supports it too.

    Scoring uses neg_mean_squared_error since GridSearchCV maximizes;
    we convert to RMSE for reporting. Final model selection uses log loss
    computed separately after CV.
    """
    if param_grid is None:
        log.info("  %s: no grid search (baseline model)", name)
        try:
            pipeline.fit(X_train, y_train, model__sample_weight=weights_train)
        except TypeError:
            pipeline.fit(X_train, y_train)
        return pipeline, pd.DataFrame()

    log.info("  %s: grid searching %d combinations ...",
             name, _grid_size(param_grid))

    # Build fit_params for sample_weight passthrough
    fit_params = {}
    try:
        # Test if the model accepts sample_weight
        test_pipe = pipeline.__class__(
            **{k: v for k, v in pipeline.get_params().items()
               if "__" not in k}
        )
        fit_params["model__sample_weight"] = weights_train.values
    except Exception:
        pass

    # Always try to pass sample_weight — GridSearchCV will ignore if unsupported
    fit_params["model__sample_weight"] = weights_train.values

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_split,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
        error_score="raise",
    )

    try:
        gs.fit(X_train, y_train, **fit_params)
    except TypeError:
        # Model doesn't accept sample_weight — fit without it
        log.warning("  %s: sample_weight not supported, fitting without weights", name)
        gs.fit(X_train, y_train)

    best_params = gs.best_params_
    best_score  = np.sqrt(-gs.best_score_)  # RMSE

    log.info("  %s: best RMSE=%.4f | params=%s", name, best_score, best_params)

    results_df = pd.DataFrame(gs.cv_results_)
    results_df["model"] = name
    results_df["rmse"]  = np.sqrt(-results_df["mean_test_score"])

    return gs.best_estimator_, results_df


def _grid_size(param_grid: dict) -> int:
    size = 1
    for v in param_grid.values():
        size *= len(v)
    return size


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    pipeline,
    folds: list[tuple],
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
) -> dict:
    """Evaluates a fitted pipeline across all time-series CV folds."""
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train         = weights.iloc[train_idx]

        try:
            pipeline.fit(X_train, y_train, model__sample_weight=w_train)
        except TypeError:
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        cls    = regression_to_classification_metrics(y_test.values, y_pred)

        fold_metrics.append({"fold": fold, "mae": mae, "rmse": rmse, **cls})

    fold_df = pd.DataFrame(fold_metrics)
    result  = {
        "model":         name,
        "mae_mean":      fold_df["mae"].mean(),
        "mae_std":       fold_df["mae"].std(),
        "rmse_mean":     fold_df["rmse"].mean(),
        "rmse_std":      fold_df["rmse"].std(),
        "log_loss_mean": fold_df["log_loss"].mean(),
        "log_loss_std":  fold_df["log_loss"].std(),
        "brier_mean":    fold_df["brier_score"].mean(),
        "brier_std":     fold_df["brier_score"].std(),
        "roc_auc_mean":  fold_df["roc_auc"].mean(),
        "roc_auc_std":   fold_df["roc_auc"].std(),
        "accuracy_mean": fold_df["accuracy"].mean(),
    }

    log.info(
        "  %-22s  MAE=%.2f  RMSE=%.2f  LogLoss=%.4f  AUC=%.4f  Acc=%.3f",
        name,
        result["mae_mean"],    result["rmse_mean"],
        result["log_loss_mean"], result["roc_auc_mean"],
        result["accuracy_mean"],
    )
    return result


# ---------------------------------------------------------------------------
# Tournament holdout
# ---------------------------------------------------------------------------

def tournament_holdout_eval(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    df_full: pd.DataFrame,
    best_pipeline,
    target_season: int,
) -> dict:
    log.info("Running tournament holdout eval (season=%d) ...", target_season)

    train_mask  = df_full["season"] < target_season
    tourney_mask = is_tournament_game(df_full)
    test_mask   = (df_full["season"] == target_season) & tourney_mask

    if test_mask.sum() == 0:
        log.warning(
            "No tournament games found for season %d holdout. "
            "This is expected if the season has no March/April games in the dataset.",
            target_season
        )
        return {}

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    w_train         = weights[train_mask]

    try:
        best_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    except TypeError:
        best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    cls    = regression_to_classification_metrics(y_test.values, y_pred)

    log.info(
        "Tournament holdout (n=%d) — MAE=%.2f  AUC=%.4f  Acc=%.3f",
        test_mask.sum(), mean_absolute_error(y_test, y_pred),
        cls["roc_auc"], cls["accuracy"],
    )
    return {"mae": mean_absolute_error(y_test, y_pred), **cls, "n_games": int(test_mask.sum())}


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def extract_feature_importance(
    pipeline,
    feature_names: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        imp  = pd.DataFrame({
            "feature":     feature_names,
            "importance":  np.abs(coef),
            "coefficient": coef,
        })
    elif hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature":     feature_names,
            "importance":  model.feature_importances_,
            "coefficient": model.feature_importances_,
        })
    else:
        log.info("Computing permutation importance ...")
        result = permutation_importance(
            pipeline, X_val, y_val, n_repeats=10, random_state=42
        )
        imp = pd.DataFrame({
            "feature":     feature_names,
            "importance":  result.importances_mean,
            "coefficient": result.importances_mean,
        })

    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_evaluation_plots(
    results_df: pd.DataFrame,
    best_name: str,
    best_pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_imp: pd.DataFrame,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Model Evaluation — Best: {best_name}  |  Tournament games upweighted",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    models = results_df["model"]
    x      = np.arange(len(models))
    colors = ["#2ecc71" if m == best_name else "#3498db" for m in models]

    # 1. Log loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(x, results_df["log_loss_mean"], xerr=results_df["log_loss_std"],
             color=colors, capsize=4, height=0.55)
    ax1.set_yticks(x); ax1.set_yticklabels(models, fontsize=8)
    ax1.set_xlabel("Log Loss (lower = better)")
    ax1.set_title("CV Log Loss by Model")
    ax1.invert_xaxis()

    # 2. AUC
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(x, results_df["roc_auc_mean"], xerr=results_df["roc_auc_std"],
             color=colors, capsize=4, height=0.55)
    ax2.set_yticks(x); ax2.set_yticklabels(models, fontsize=8)
    ax2.set_xlabel("ROC AUC (higher = better)")
    ax2.set_title("CV ROC AUC by Model")

    # 3. Residuals
    ax3    = fig.add_subplot(gs[0, 2])
    y_pred = best_pipeline.predict(X_val)
    ax3.scatter(y_pred, y_val.values - y_pred, alpha=0.25, s=8, color="#3498db")
    ax3.axhline(0, color="red", linewidth=1)
    ax3.set_xlabel("Predicted Spread"); ax3.set_ylabel("Residual")
    ax3.set_title(f"Residuals — {best_name}")

    # 4. Predicted vs actual
    ax4  = fig.add_subplot(gs[1, 0])
    lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
    ax4.scatter(y_val.values, y_pred, alpha=0.25, s=8, color="#9b59b6")
    ax4.plot(lims, lims, "r--", linewidth=1)
    ax4.set_xlabel("Actual Spread"); ax4.set_ylabel("Predicted Spread")
    ax4.set_title("Predicted vs Actual")

    # 5. Calibration
    ax5    = fig.add_subplot(gs[1, 1])
    y_prob = spread_to_prob(y_pred)
    y_bin  = (y_val.values > 0).astype(int)
    bins   = np.linspace(0, 1, 11)
    bcs, ars = [], []
    for i in range(len(bins) - 1):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.sum() > 0:
            bcs.append(y_prob[m].mean()); ars.append(y_bin[m].mean())
    ax5.plot(bcs, ars, "o-", color="#e74c3c", label="Model")
    ax5.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax5.set_xlabel("Predicted Win Prob"); ax5.set_ylabel("Actual Win Rate")
    ax5.set_title("Calibration Curve"); ax5.legend(fontsize=8)

    # 6. Feature importance — orange = conference features
    ax6   = fig.add_subplot(gs[1, 2])
    top_n = feature_imp.head(20)
    bar_colors = []
    for f in top_n["feature"]:
        if any(cf in f for cf in CONF_BASE_FEATURES):
            bar_colors.append("#f39c12")
        elif top_n.loc[top_n["feature"] == f, "coefficient"].values[0] >= 0:
            bar_colors.append("#e74c3c")
        else:
            bar_colors.append("#3498db")
    ax6.barh(range(len(top_n)), top_n["importance"], color=bar_colors, height=0.6)
    ax6.set_yticks(range(len(top_n)))
    ax6.set_yticklabels(top_n["feature"], fontsize=7)
    ax6.set_xlabel("Importance / |Coefficient|")
    ax6.set_title("Top 20 Features\n(orange = conference feature)")
    ax6.invert_yaxis()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Plots saved -> %s", output_path)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="March Madness model training with grid search")
    parser.add_argument("--data-dir",          default="data")
    parser.add_argument("--model-dir",         default="models")
    parser.add_argument("--target-season",     type=int,   default=2024)
    parser.add_argument("--cv-splits",         type=int,   default=5)
    parser.add_argument("--tournament-weight", type=float, default=3.0,
                        help="Sample weight multiplier for tournament games (default: 3.0)")
    parser.add_argument("--n-jobs",            type=int,   default=-1,
                        help="Parallel jobs for grid search (-1 = all cores)")
    return parser.parse_args()


def main():
    args      = parse_args()
    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and prep
    # ------------------------------------------------------------------
    df, tf, raw_games = load_data(data_dir)
    df = df.sort_values("date").reset_index(drop=True)

    X, y, weights = prepare_features(
        df, tf, raw_games,
        target_col="point_diff",
        tournament_weight=args.tournament_weight,
    )
    feature_names = X.columns.tolist()

    # ------------------------------------------------------------------
    # Build time-series CV folds
    # ------------------------------------------------------------------
    seasons = df["season"].values
    folds   = build_ts_folds(seasons, n_splits=args.cv_splits)
    cv_split = folds_to_predefined_split(len(X), folds)

    # Training data for grid search = all seasons before target
    train_mask       = df["season"] < args.target_season
    X_train, y_train = X[train_mask], y[train_mask]
    w_train          = weights[train_mask]
    X_val,   y_val   = X[~train_mask], y[~train_mask]

    # Rebuild folds on training subset for grid search
    train_seasons  = df["season"].values[train_mask]
    train_folds    = build_ts_folds(train_seasons, n_splits=args.cv_splits)
    train_cv_split = folds_to_predefined_split(len(X_train), train_folds)

    # ------------------------------------------------------------------
    # Grid search + evaluation
    # ------------------------------------------------------------------
    model_configs    = get_model_configs()
    all_gs_results   = []
    eval_results     = []
    tuned_pipelines  = {}

    log.info("=" * 60)
    log.info("GRID SEARCH  (tournament weight=%.1fx)", args.tournament_weight)
    log.info("=" * 60)

    for name, config in model_configs.items():
        log.info("-> %s", name)
        best_pipe, gs_df = run_grid_search(
            name          = name,
            pipeline      = config["pipeline"],
            param_grid    = config["param_grid"],
            X_train       = X_train,
            y_train       = y_train,
            weights_train = w_train,
            cv_split      = train_cv_split,
            n_jobs        = args.n_jobs,
        )
        tuned_pipelines[name] = best_pipe
        if not gs_df.empty:
            all_gs_results.append(gs_df)

    log.info("=" * 60)
    log.info("CROSS-VALIDATION EVALUATION")
    log.info("=" * 60)

    for name, pipeline in tuned_pipelines.items():
        result = evaluate_model(name, pipeline, folds, X, y, weights)
        eval_results.append(result)

    results_df = pd.DataFrame(eval_results).sort_values("log_loss_mean")
    best_name  = results_df.iloc[0]["model"]

    log.info("Best model by CV log loss: %s", best_name)

    print("\n" + "=" * 75)
    print("MODEL COMPARISON (sorted by log loss)")
    print("=" * 75)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("=" * 75 + "\n")

    # ------------------------------------------------------------------
    # Refit best model on full training data
    # ------------------------------------------------------------------
    best_pipeline = tuned_pipelines[best_name]
    try:
        best_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    except TypeError:
        best_pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Tournament holdout
    # ------------------------------------------------------------------
    holdout = tournament_holdout_eval(
        X, y, weights, df, best_pipeline, args.target_season
    )
    if holdout:
        print("TOURNAMENT HOLDOUT METRICS")
        print("=" * 45)
        for k, v in holdout.items():
            print(f"  {k:<18} {v:.4f}" if isinstance(v, float) else f"  {k:<18} {v}")
        print()

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    feature_imp = extract_feature_importance(best_pipeline, feature_names, X_val, y_val)
    print("TOP 20 FEATURES")
    print("=" * 60)
    print(
        feature_imp.head(20)[["feature", "importance", "coefficient"]]
        .to_string(index=False, float_format="%.4f")
    )
    print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results_df.to_csv(model_dir / "model_comparison.csv", index=False)
    feature_imp.to_csv(model_dir / "feature_importance.csv", index=False)
    joblib.dump(best_pipeline, model_dir / "best_model.joblib")

    if all_gs_results:
        gs_combined = pd.concat(all_gs_results, ignore_index=True)
        gs_combined.to_csv(model_dir / "grid_search_results.csv", index=False)
        log.info("Grid search results -> %s/grid_search_results.csv", args.model_dir)

    make_evaluation_plots(
        results_df, best_name, best_pipeline,
        X_val, y_val, feature_imp,
        model_dir / "evaluation_plots.png",
    )

    log.info("Best model saved -> %s/best_model.joblib", args.model_dir)
    log.info(
        "Next step: python predict_bracket.py --model-dir %s --data-dir %s",
        args.model_dir, args.data_dir,
    )


if __name__ == "__main__":
    main()