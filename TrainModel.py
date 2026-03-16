"""
train_model.py
--------------
Model selection and evaluation for March Madness spread prediction.

Approach:
  - Target: point_diff (home_score - away_score) via regression
  - Win probability derived from predicted spread via logistic conversion
  - Tournament games upweighted during training
  - Conference-level features computed from game data (pace, strength, depth)
  - Models evaluated: Linear, Ridge, Lasso, ElasticNet, Gradient Boosting
  - Validation: time-series cross-validation (train on older seasons, test on recent)

Outputs (models/ directory):
  - best_model.joblib
  - model_comparison.csv
  - feature_importance.csv
  - evaluation_plots.png

Usage:
  python train_model.py [--data-dir data] [--model-dir models] [--target-season 2024]
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    log_loss, brier_score_loss, roc_auc_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPREAD_SCALE       = 10.0
TOURNAMENT_WEIGHT  = 3.0
TOURNAMENT_KEYWORDS = [
    "NCAA", "Tournament", "First Round", "Second Round",
    "Sweet 16", "Elite 8", "Final Four", "Championship"
]

# ---------------------------------------------------------------------------
# Feature allowlist — pre-game knowable team-level features only
# ---------------------------------------------------------------------------
# For each base feature the game dataset contains home_<f>, away_<f>, diff_<f>.
# Conference features are added separately below in build_conference_features().
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    # Shooting efficiency rates
    "fg_pct",
    "fg3_pct",
    "ft_pct",

    # Season record
    "win_percentage",
    "conference_win_percentage",
    "home_wins",
    "home_losses",
    "away_wins",
    "away_losses",
    "conference_wins",
    "conference_losses",
    "wins",
    "losses",

    # Poll rankings
    "ap_rank",
    "coach_rank",
    "is_ranked",

    # Tournament seeding
    "playoff_seed",
]

# Conference features are prefixed with conf_ and get the same
# home_ / away_ / diff_ expansion applied after computation.
CONF_BASE_FEATURES = [
    "conf_pace",       # avg total pts/game in conference games
    "conf_strength",   # non-conf win% of conference teams
    "conf_depth",      # avg absolute margin in conference games (lower = more competitive)
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


def is_tournament_game(period_detail: pd.Series) -> pd.Series:
    mask = pd.Series(False, index=period_detail.index)
    for kw in TOURNAMENT_KEYWORDS:
        mask |= period_detail.fillna("").str.contains(kw, case=False)
    return mask


def regression_to_classification_metrics(
    y_true_spread: np.ndarray,
    y_pred_spread: np.ndarray,
    scale: float = SPREAD_SCALE,
) -> dict:
    y_prob        = spread_to_prob(y_pred_spread, scale)
    y_true_binary = (y_true_spread > 0).astype(int)
    return {
        "log_loss":    log_loss(y_true_binary, y_prob),
        "brier_score": brier_score_loss(y_true_binary, y_prob),
        "roc_auc":     roc_auc_score(y_true_binary, y_prob),
        "accuracy":    ((y_prob > 0.5) == y_true_binary).mean(),
    }


# ---------------------------------------------------------------------------
# Conference feature computation
# ---------------------------------------------------------------------------

def build_conference_features(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes three conference-level features per (conference_id, season):

      conf_pace     — average total points per game among conference games.
                      Higher = faster / more up-tempo conference.

      conf_strength — win% of conference teams in non-conference games only.
                      Isolates schedule strength from intra-conference results.

      conf_depth    — average absolute point margin in conference games.
                      Lower = more competitive top-to-bottom; higher = more top-heavy.

    These are joined onto team_features and then expanded into
    home_/away_/diff_ columns the same way as all other features.

    Parameters
    ----------
    games : DataFrame
        Raw game_dataset.parquet — needs home_team_id, away_team_id,
        home_score, away_score, season, and point_diff columns.
    team_features : DataFrame
        team_features.parquet — needs team_id, season, conference_id columns
        so we can map each team to their conference.

    Returns
    -------
    team_features with three new columns: conf_pace, conf_strength, conf_depth.
    """
    log.info("Computing conference features ...")

    # We need conference_id for each team — pull from team_features
    conf_map = (
        team_features[["team_id", "season", "conference_id"]]
        .dropna(subset=["conference_id"])
        .drop_duplicates()
    )

    if conf_map.empty or "conference_id" not in team_features.columns:
        log.warning(
            "conference_id not available in team_features — "
            "skipping conference features. "
            "Re-run collect_data.py to populate standings data."
        )
        return team_features

    # Join conference_id onto both sides of each game
    g = games.copy()
    g = g.merge(
        conf_map.rename(columns={"team_id": "home_team_id", "conference_id": "home_conf_id"}),
        on=["home_team_id", "season"], how="left"
    )
    g = g.merge(
        conf_map.rename(columns={"team_id": "away_team_id", "conference_id": "away_conf_id"}),
        on=["away_team_id", "season"], how="left"
    )

    # Flag intra-conference games (both teams same conference)
    g["is_conf_game"] = g["home_conf_id"] == g["away_conf_id"]
    g["total_score"]  = g["home_score"] + g["away_score"]
    g["abs_margin"]   = g["point_diff"].abs()

    conf_games    = g[g["is_conf_game"]].copy()
    nonconf_games = g[~g["is_conf_game"]].copy()

    # ------------------------------------------------------------------
    # 1. Pace proxy — avg total pts/game in conference games
    #    Use home_conf_id since both sides are same conference in conf games
    # ------------------------------------------------------------------
    pace = (
        conf_games
        .groupby(["home_conf_id", "season"])["total_score"]
        .mean()
        .reset_index()
        .rename(columns={"home_conf_id": "conference_id", "total_score": "conf_pace"})
    )

    # ------------------------------------------------------------------
    # 2. Conference strength — non-conf win% per conference per season
    #    Each non-conf game contributes one win and one loss observation
    # ------------------------------------------------------------------
    # Home team perspective
    home_nonconf = nonconf_games[["home_conf_id", "season", "home_score", "away_score"]].copy()
    home_nonconf["win"] = (home_nonconf["home_score"] > home_nonconf["away_score"]).astype(int)
    home_nonconf = home_nonconf.rename(columns={"home_conf_id": "conference_id"})

    # Away team perspective
    away_nonconf = nonconf_games[["away_conf_id", "season", "home_score", "away_score"]].copy()
    away_nonconf["win"] = (away_nonconf["away_score"] > away_nonconf["home_score"]).astype(int)
    away_nonconf = away_nonconf.rename(columns={"away_conf_id": "conference_id"})

    all_nonconf = pd.concat(
        [home_nonconf[["conference_id", "season", "win"]],
         away_nonconf[["conference_id", "season", "win"]]],
        ignore_index=True
    )
    strength = (
        all_nonconf
        .groupby(["conference_id", "season"])["win"]
        .mean()
        .reset_index()
        .rename(columns={"win": "conf_strength"})
    )

    # ------------------------------------------------------------------
    # 3. Competitive depth — avg absolute margin in conference games
    # ------------------------------------------------------------------
    depth = (
        conf_games
        .groupby(["home_conf_id", "season"])["abs_margin"]
        .mean()
        .reset_index()
        .rename(columns={"home_conf_id": "conference_id", "abs_margin": "conf_depth"})
    )

    # ------------------------------------------------------------------
    # Merge all three back into team_features
    # ------------------------------------------------------------------
    conf_stats = (
        pace
        .merge(strength, on=["conference_id", "season"], how="outer")
        .merge(depth,    on=["conference_id", "season"], how="outer")
    )

    tf = team_features.merge(conf_stats, on=["conference_id", "season"], how="left")

    n_with_conf = tf["conf_pace"].notna().sum()
    log.info(
        "  Conference features computed — "
        "%d / %d team-season records have conference data",
        n_with_conf, len(tf)
    )
    log.info(
        "  conf_pace range:     %.1f – %.1f pts/game",
        tf["conf_pace"].min(), tf["conf_pace"].max()
    )
    log.info(
        "  conf_strength range: %.3f – %.3f non-conf win%%",
        tf["conf_strength"].min(), tf["conf_strength"].max()
    )
    log.info(
        "  conf_depth range:    %.1f – %.1f avg margin",
        tf["conf_depth"].min(), tf["conf_depth"].max()
    )

    return tf


# ---------------------------------------------------------------------------
# Data loading and feature preparation
# ---------------------------------------------------------------------------

def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (game_dataset, team_features).
    team_features is loaded separately so conference features can be
    computed before joining onto games.
    """
    game_fp = data_dir / "game_dataset.parquet"
    tf_fp   = data_dir / "team_features.parquet"
    raw_fp  = data_dir / "games.parquet"

    if not game_fp.exists():
        raise FileNotFoundError(
            f"game_dataset.parquet not found in {data_dir}. "
            "Run collect_data.py first."
        )

    df = pd.read_parquet(game_fp)
    log.info("Loaded game dataset: %d rows, %d cols", *df.shape)

    # Load team_features and raw games for conference computation
    tf = pd.read_parquet(tf_fp)   if tf_fp.exists()  else pd.DataFrame()
    raw_games = pd.read_parquet(raw_fp) if raw_fp.exists() else pd.DataFrame()

    return df, tf, raw_games


def prepare_features(
    df: pd.DataFrame,
    tf: pd.DataFrame,
    raw_games: pd.DataFrame,
    target_col: str = "point_diff",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Builds the final feature matrix using:
      1. Explicit allowlist of pre-game team features
      2. Computed conference-level features joined onto teams then games
    """
    df = df.copy()

    # Sample weights
    if "period_detail" in df.columns:
        tourney_flag = is_tournament_game(df["period_detail"])
    else:
        tourney_flag = pd.Series(False, index=df.index)

    weights        = np.where(tourney_flag, TOURNAMENT_WEIGHT, 1.0)
    sample_weights = pd.Series(weights, index=df.index)

    # ------------------------------------------------------------------
    # Conference features
    # ------------------------------------------------------------------
    if not tf.empty and not raw_games.empty:
        tf_with_conf = build_conference_features(raw_games, tf)

        # Join conference features onto game_dataset for home and away teams
        conf_cols = ["team_id", "season"] + CONF_BASE_FEATURES
        conf_cols = [c for c in conf_cols if c in tf_with_conf.columns]

        home_conf = (
            tf_with_conf[conf_cols]
            .rename(columns={"team_id": "home_team_id"})
            .rename(columns={f: f"home_{f}" for f in CONF_BASE_FEATURES if f in tf_with_conf.columns})
        )
        away_conf = (
            tf_with_conf[conf_cols]
            .rename(columns={"team_id": "away_team_id"})
            .rename(columns={f: f"away_{f}" for f in CONF_BASE_FEATURES if f in tf_with_conf.columns})
        )

        df = df.merge(home_conf, on=["home_team_id", "season"], how="left")
        df = df.merge(away_conf, on=["away_team_id", "season"], how="left")

        # Differential columns for conference features
        for f in CONF_BASE_FEATURES:
            h, a = f"home_{f}", f"away_{f}"
            if h in df.columns and a in df.columns:
                df[f"diff_{f}"] = df[h] - df[a]
    else:
        log.warning("team_features or raw games not available — skipping conference features")

    # ------------------------------------------------------------------
    # Apply allowlist (team + conference features)
    # ------------------------------------------------------------------
    all_base = BASE_FEATURES + CONF_BASE_FEATURES
    allowed_cols  = expand_allowlist(all_base)
    available_cols = [c for c in allowed_cols if c in df.columns]
    missing_cols   = [c for c in allowed_cols if c not in df.columns]

    if missing_cols:
        log.warning(
            "%d allowlist features not found in dataset (skipping): %s",
            len(missing_cols), missing_cols
        )

    X = df[available_cols].copy()

    # Drop columns with >60% missing values
    missing_frac = X.isnull().mean()
    high_missing = missing_frac[missing_frac >= 0.6].index.tolist()
    if high_missing:
        log.warning(
            "Dropping %d cols with >60%% missing: %s",
            len(high_missing), high_missing
        )
        X = X.drop(columns=high_missing)

    y = df[target_col]

    log.info(
        "Feature matrix: %d rows × %d cols | "
        "Tournament games: %d (weighted %.1fx)",
        *X.shape, tourney_flag.sum(), TOURNAMENT_WEIGHT,
    )
    log.info("Features used: %s", X.columns.tolist())

    return X, y, sample_weights


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_pipelines() -> dict:
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    return {
        "Linear Regression": Pipeline([
            ("pre",   preprocessor),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("pre",   preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Lasso": Pipeline([
            ("pre",   preprocessor),
            ("model", Lasso(alpha=0.1, max_iter=5000)),
        ]),
        "ElasticNet": Pipeline([
            ("pre",   preprocessor),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ]),
        "Gradient Boosting": Pipeline([
            ("pre",   preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    df_full: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    log.info("Running time-series CV (%d splits) ...", n_splits)

    seasons        = df_full["season"].values
    unique_seasons = np.sort(np.unique(seasons))

    if len(unique_seasons) < n_splits + 1:
        n_splits = len(unique_seasons) - 1
        log.warning("Reduced CV splits to %d based on available seasons", n_splits)

    fold_indices = []
    for i in range(n_splits):
        cutoff    = unique_seasons[-(n_splits - i)]
        train_idx = np.where(seasons < cutoff)[0]
        test_idx  = np.where(seasons == cutoff)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            fold_indices.append((train_idx, test_idx))

    pipelines = build_pipelines()
    results   = []

    for name, pipeline in pipelines.items():
        log.info("  Evaluating: %s", name)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(fold_indices):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train         = weights.iloc[train_idx]

            try:
                pipeline.fit(X_train, y_train, model__sample_weight=w_train)
            except TypeError:
                pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cls  = regression_to_classification_metrics(y_test.values, y_pred)

            fold_metrics.append({"fold": fold, "mae": mae, "rmse": rmse, **cls})

        fold_df = pd.DataFrame(fold_metrics)
        results.append({
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
        })
        log.info(
            "    MAE=%.2f  RMSE=%.2f  LogLoss=%.4f  AUC=%.4f  Acc=%.3f",
            results[-1]["mae_mean"],   results[-1]["rmse_mean"],
            results[-1]["log_loss_mean"], results[-1]["roc_auc_mean"],
            results[-1]["accuracy_mean"],
        )

    return pd.DataFrame(results).sort_values("log_loss_mean")


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

    train_mask = df_full["season"] < target_season
    test_mask  = (
        (df_full["season"] == target_season) &
        is_tournament_game(
            df_full.get("period_detail", pd.Series("", index=df_full.index))
        )
    )

    if test_mask.sum() == 0:
        log.warning(
            "No tournament games found for season %d — skipping holdout.",
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
        "Tournament holdout — MAE=%.2f  AUC=%.4f  Acc=%.3f  n=%d",
        mean_absolute_error(y_test, y_pred),
        cls["roc_auc"], cls["accuracy"], test_mask.sum(),
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
        imp = pd.DataFrame({
            "feature":     feature_names,
            "importance":  np.abs(model.coef_),
            "coefficient": model.coef_,
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
    fig.suptitle(f"Model Evaluation — Best: {best_name}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    models = results_df["model"]
    x      = np.arange(len(models))
    colors = lambda m: ["#2ecc71" if n == best_name else "#3498db" for n in m]

    # 1. Log loss comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(x, results_df["log_loss_mean"], xerr=results_df["log_loss_std"],
             color=colors(models), capsize=4, height=0.5)
    ax1.set_yticks(x); ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel("Log Loss (lower = better)")
    ax1.set_title("CV Log Loss by Model")
    ax1.invert_xaxis()

    # 2. AUC comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(x, results_df["roc_auc_mean"], xerr=results_df["roc_auc_std"],
             color=colors(models), capsize=4, height=0.5)
    ax2.set_yticks(x); ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel("ROC AUC (higher = better)")
    ax2.set_title("CV ROC AUC by Model")

    # 3. Residuals
    ax3    = fig.add_subplot(gs[0, 2])
    y_pred = best_pipeline.predict(X_val)
    ax3.scatter(y_pred, y_val.values - y_pred, alpha=0.3, s=10, color="#3498db")
    ax3.axhline(0, color="red", linewidth=1)
    ax3.set_xlabel("Predicted Spread"); ax3.set_ylabel("Residual")
    ax3.set_title(f"Residuals — {best_name}")

    # 4. Predicted vs actual
    ax4  = fig.add_subplot(gs[1, 0])
    lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
    ax4.scatter(y_val.values, y_pred, alpha=0.3, s=10, color="#9b59b6")
    ax4.plot(lims, lims, "r--", linewidth=1)
    ax4.set_xlabel("Actual Spread"); ax4.set_ylabel("Predicted Spread")
    ax4.set_title("Predicted vs Actual")

    # 5. Calibration curve
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

    # 6. Feature importance — highlight conference features in orange
    ax6   = fig.add_subplot(gs[1, 2])
    top_n = feature_imp.head(20)
    bar_colors = []
    for f in top_n["feature"]:
        if any(cf in f for cf in CONF_BASE_FEATURES):
            bar_colors.append("#f39c12")   # orange = conference feature
        elif top_n.loc[top_n["feature"] == f, "coefficient"].values[0] >= 0:
            bar_colors.append("#e74c3c")   # red = positive coefficient
        else:
            bar_colors.append("#3498db")   # blue = negative coefficient
    ax6.barh(range(len(top_n)), top_n["importance"], color=bar_colors, height=0.6)
    ax6.set_yticks(range(len(top_n)))
    ax6.set_yticklabels(top_n["feature"], fontsize=7)
    ax6.set_xlabel("Importance / |Coefficient|")
    ax6.set_title("Top 20 Features\n(orange = conference feature)")
    ax6.invert_yaxis()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Saved evaluation plots → %s", output_path)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="March Madness model selection")
    parser.add_argument("--data-dir",      default="data",   help="Directory with parquet files")
    parser.add_argument("--model-dir",     default="models", help="Directory to save outputs")
    parser.add_argument("--target-season", type=int, default=2024,
                        help="Most recent complete season — used as holdout")
    parser.add_argument("--cv-splits",     type=int, default=5)
    return parser.parse_args()


def main():
    args      = parse_args()
    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df, tf, raw_games = load_data(data_dir)
    df = df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Feature prep (includes conference feature computation)
    # ------------------------------------------------------------------
    X, y, weights = prepare_features(df, tf, raw_games, target_col="point_diff")
    feature_names = X.columns.tolist()

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    results_df = run_cv(X, y, weights, df, n_splits=args.cv_splits)
    best_name  = results_df.iloc[0]["model"]

    log.info("Best model by CV log loss: %s", best_name)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by log loss)")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Fit best model on training seasons
    # ------------------------------------------------------------------
    pipelines     = build_pipelines()
    best_pipeline = pipelines[best_name]

    train_mask       = df["season"] < args.target_season
    X_train, y_train = X[train_mask], y[train_mask]
    w_train          = weights[train_mask]
    X_val,   y_val   = X[~train_mask], y[~train_mask]

    try:
        best_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    except TypeError:
        best_pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Tournament holdout
    # ------------------------------------------------------------------
    holdout = tournament_holdout_eval(X, y, weights, df, best_pipeline, args.target_season)
    if holdout:
        print("TOURNAMENT HOLDOUT METRICS")
        print("=" * 40)
        for k, v in holdout.items():
            print(f"  {k:<15} {v:.4f}" if isinstance(v, float) else f"  {k:<15} {v}")
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
    results_df.to_csv(model_dir  / "model_comparison.csv",   index=False)
    feature_imp.to_csv(model_dir / "feature_importance.csv", index=False)
    joblib.dump(best_pipeline,    model_dir / "best_model.joblib")

    make_evaluation_plots(
        results_df, best_name, best_pipeline,
        X_val, y_val, feature_imp,
        model_dir / "evaluation_plots.png",
    )

    log.info("Best model saved → %s/best_model.joblib", args.model_dir)
    log.info(
        "Next step: python predict_bracket.py --model-dir %s --data-dir %s",
        args.model_dir, args.data_dir
    )


if __name__ == "__main__":
    main()
