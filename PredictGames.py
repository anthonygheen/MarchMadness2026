"""
predict_games.py
----------------
Pulls live NCAA Tournament games from the BallDontLie API and generates
per-game predictions using the trained model.

For each game:
  - Win probability for each team
  - Predicted point spread
  - Predicted total score
  - Confidence tier (Lock / Lean / Toss-Up)

Confidence tiers:
  Lock     — win probability >= 75%
  Lean     — win probability 60-74%
  Toss-Up  — win probability < 60%

Outputs:
  - Terminal prediction report
  - models/predictions_MMDD.csv     (overwrites if exists)
  - docs/predictions_MMDD.md        (GitHub-renderable markdown)

Usage:
  # Today's games (auto date)
  python predict_games.py --season 2025 --today

  # Specific date
  python predict_games.py --season 2025 --date 2026-03-20

  # Specific round
  python predict_games.py --season 2025 --round 1

  # Live games only
  python predict_games.py --season 2025 --status in --today

  # Completed games (check results)
  python predict_games.py --season 2025 --status post --round 1
"""

import os
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import joblib
from dotenv import load_dotenv
from scipy.special import expit

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL         = "https://api.balldontlie.io/ncaab/v1"
SPREAD_SCALE     = 10.0
RATE_LIMIT_SLEEP = 0.12

BASE_FEATURES = [
    "fg_pct", "fg3_pct", "ft_pct",
    "win_percentage", "conference_win_percentage",
    "home_wins", "home_losses", "away_wins", "away_losses",
    "conference_wins", "conference_losses",
    "ap_rank", "coach_rank", "is_ranked",
    "playoff_seed",
]
CONF_BASE_FEATURES = [
    "conf_pace",
    "conf_strength",
    "conf_depth",
]

LOCK_THRESHOLD = 0.75
LEAN_THRESHOLD = 0.60

ROUND_NAMES = {
    0: "First Four",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}

CONFIDENCE_ICONS = {
    "LOCK":    "🔒",
    "LEAN":    "📊",
    "TOSS-UP": "🪙",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_headers() -> dict:
    api_key = os.environ.get("BDL_API_KEY")
    if not api_key:
        raise EnvironmentError("Set BDL_API_KEY environment variable.")
    return {"Authorization": api_key}


# ---------------------------------------------------------------------------
# Pull tournament games
# ---------------------------------------------------------------------------

def pull_tournament_games(
    season: int,
    round_filter: int | None = None,
    date_filter: str | None = None,
    status_filter: str | None = None,
) -> pd.DataFrame:
    """
    Pulls games from the bracket endpoint for the given season.

    Round numbers:
      0 = First Four,  1 = Round of 64,  2 = Round of 32
      3 = Sweet 16,    4 = Elite 8,      5 = Final Four,  6 = Championship

    Status: 'pre' (upcoming), 'in' (live), 'post' (completed)
    """
    log.info("Pulling tournament games for season %d ...", season)

    records = []
    params  = {"season": season, "per_page": 100}
    cursor  = None

    while True:
        if cursor:
            params["cursor"] = cursor
        for attempt in range(5):
            resp = requests.get(
                f"{BASE_URL}/bracket",
                headers=get_headers(),
                params=params,
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        records.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(RATE_LIMIT_SLEEP)

    if not records:
        raise RuntimeError(f"No bracket data found for season {season}.")

    rows = []
    for entry in records:
        home = entry.get("home_team") or {}
        away = entry.get("away_team") or {}
        if not home.get("id") or not away.get("id"):
            continue
        rows.append({
            "game_id":       entry.get("game_id"),
            "round":         entry.get("round"),
            "bracket_loc":   entry.get("bracket_location"),
            "date":          entry.get("date"),
            "location":      entry.get("location"),
            "status":        entry.get("status"),
            "status_detail": entry.get("status_detail"),
            "home_team_id":  home.get("id"),
            "home_team":     home.get("full_name") or home.get("name"),
            "home_seed":     home.get("seed"),
            "home_score":    home.get("score"),
            "home_winner":   home.get("winner"),
            "away_team_id":  away.get("id"),
            "away_team":     away.get("full_name") or away.get("name"),
            "away_seed":     away.get("seed"),
            "away_score":    away.get("score"),
            "away_winner":   away.get("winner"),
        })

    df = pd.DataFrame(rows)
    df["home_seed"]   = pd.to_numeric(df["home_seed"],   errors="coerce")
    df["away_seed"]   = pd.to_numeric(df["away_seed"],   errors="coerce")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

    if round_filter is not None:
        df = df[df["round"] == round_filter]
        log.info("  Filtered to round %d: %d games", round_filter, len(df))

    if date_filter:
        df = df[df["date_parsed"].dt.strftime("%Y-%m-%d") == date_filter]
        log.info("  Filtered to date %s: %d games", date_filter, len(df))

    if status_filter:
        df = df[df["status"] == status_filter]
        log.info("  Filtered to status '%s': %d games", status_filter, len(df))

    log.info("Games to predict: %d", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team features
# ---------------------------------------------------------------------------

def load_team_features(
    data_dir: Path,
    season: int,
    team_ids: list[int],
) -> pd.DataFrame:
    tf_path  = data_dir / "team_features.parquet"
    raw_path = data_dir / "games.parquet"

    if not tf_path.exists():
        raise FileNotFoundError(
            f"team_features.parquet not found in {data_dir}. "
            "Run collect_data.py first."
        )

    tf        = pd.read_parquet(tf_path)
    raw_games = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()

    tf_season = tf[tf["season"] == season].copy()
    if tf_season.empty:
        raise RuntimeError(
            f"No features for season {season}. "
            f"Run collect_data.py --seasons {season}."
        )

    if not raw_games.empty:
        tf_season = _add_conference_features(raw_games, tf_season, season)

    tf_filtered = tf_season[tf_season["team_id"].isin(team_ids)].copy()

    missing = set(team_ids) - set(tf_filtered["team_id"].tolist())
    if missing:
        log.warning("Missing features for team IDs: %s", missing)

    return tf_filtered


def _add_conference_features(
    raw_games: pd.DataFrame,
    tf: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    if "conference_id" not in tf.columns:
        return tf

    conf_map = tf[["team_id", "season", "conference_id"]].dropna().drop_duplicates()
    g = raw_games[raw_games["season"] == season].copy()
    if g.empty:
        return tf

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
    g["abs_margin"]   = (g["home_score"] - g["away_score"]).abs()

    conf_games    = g[g["is_conf_game"]]
    nonconf_games = g[~g["is_conf_game"]]

    pace = (
        conf_games.groupby("home_conf_id")["total_score"].mean().reset_index()
        .rename(columns={"home_conf_id": "conference_id", "total_score": "conf_pace"})
    )
    depth = (
        conf_games.groupby("home_conf_id")["abs_margin"].mean().reset_index()
        .rename(columns={"home_conf_id": "conference_id", "abs_margin": "conf_depth"})
    )

    home_nc = nonconf_games[["home_conf_id", "home_score", "away_score"]].copy()
    home_nc["win"] = (home_nc["home_score"] > home_nc["away_score"]).astype(int)
    home_nc = home_nc.rename(columns={"home_conf_id": "conference_id"})

    away_nc = nonconf_games[["away_conf_id", "home_score", "away_score"]].copy()
    away_nc["win"] = (away_nc["away_score"] > away_nc["home_score"]).astype(int)
    away_nc = away_nc.rename(columns={"away_conf_id": "conference_id"})

    strength = (
        pd.concat([home_nc[["conference_id", "win"]], away_nc[["conference_id", "win"]]])
        .groupby("conference_id")["win"].mean().reset_index()
        .rename(columns={"win": "conf_strength"})
    )

    conf_stats = (
        pace
        .merge(strength, on="conference_id", how="outer")
        .merge(depth,    on="conference_id", how="outer")
    )
    return tf.merge(conf_stats, on="conference_id", how="left")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def build_matchup_row(
    team_a: pd.Series,
    team_b: pd.Series,
) -> pd.DataFrame:
    all_features = BASE_FEATURES + CONF_BASE_FEATURES
    row = {}
    for f in all_features:
        ha = team_a.get(f, np.nan)
        hb = team_b.get(f, np.nan)
        row[f"home_{f}"] = ha
        row[f"away_{f}"] = hb
        try:
            row[f"diff_{f}"] = float(ha) - float(hb)
        except (TypeError, ValueError):
            row[f"diff_{f}"] = np.nan
    return pd.DataFrame([row])


def predict_game(
    model,
    team_a_features: pd.Series,
    team_b_features: pd.Series,
) -> dict:
    """
    Neutral-court prediction. Averages both orientations to remove
    the home court advantage baked into the regular-season trained model.
    """
    feat_ab = build_matchup_row(team_a_features, team_b_features)
    feat_ba = build_matchup_row(team_b_features, team_a_features)

    try:
        model_features = model.named_steps["pre"].named_steps["imputer"].feature_names_in_
        feat_ab = feat_ab.reindex(columns=model_features)
        feat_ba = feat_ba.reindex(columns=model_features)
    except AttributeError:
        pass

    spread_ab = model.predict(feat_ab)[0]
    spread_ba = model.predict(feat_ba)[0]

    prob_ab = expit( spread_ab / SPREAD_SCALE)
    prob_ba = expit(-spread_ba / SPREAD_SCALE)

    team_a_prob = float((prob_ab + prob_ba) / 2)
    team_b_prob = 1.0 - team_a_prob
    spread      = float((spread_ab - spread_ba) / 2)

    # Estimate total from conference pace
    pace_a = team_a_features.get("conf_pace", 140.0)
    pace_b = team_b_features.get("conf_pace", 140.0)
    predicted_total = float((pace_a + pace_b) / 2) if pd.notna(pace_a) and pd.notna(pace_b) else 140.0

    max_prob = max(team_a_prob, team_b_prob)
    if max_prob >= LOCK_THRESHOLD:
        confidence = "LOCK"
    elif max_prob >= LEAN_THRESHOLD:
        confidence = "LEAN"
    else:
        confidence = "TOSS-UP"

    return {
        "spread":          round(spread, 1),
        "team_a_prob":     round(team_a_prob, 4),
        "team_b_prob":     round(team_b_prob, 4),
        "predicted_total": round(predicted_total, 1),
        "confidence":      confidence,
    }


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_predictions(results: list[dict]) -> None:
    if not results:
        print("\nNo predictions to display.")
        return

    by_round = {}
    for r in results:
        rnd = r["round"]
        if rnd not in by_round:
            by_round[rnd] = []
        by_round[rnd].append(r)

    print()
    print("=" * 72)
    print("  NCAA TOURNAMENT GAME PREDICTIONS")
    print(f"  {datetime.now().strftime('%B %d, %Y  %I:%M %p')}")
    print("=" * 72)

    for rnd in sorted(by_round.keys()):
        round_name = ROUND_NAMES.get(rnd, f"Round {rnd}")
        print(f"\n  {'─' * 68}")
        print(f"  {round_name.upper()}")
        print(f"  {'─' * 68}")

        for r in by_round[rnd]:
            icon       = CONFIDENCE_ICONS.get(r["confidence"], "")
            dt         = r.get("date_parsed")
            dt_str     = dt.strftime("%a %b %d  %I:%M %p") if pd.notna(dt) else ""
            loc_str    = r.get("location") or ""
            status_str = ""
            if r["status"] == "post":
                status_str = (f"  [FINAL: {r['home_team']} {r['home_score']} - "
                              f"{r['away_score']} {r['away_team']}]")
            elif r["status"] == "in":
                status_str = f"  [LIVE: {r.get('status_detail', '')}]"

            print()
            print(f"  {icon} {r['confidence']}{status_str}")
            if dt_str or loc_str:
                print(f"  {dt_str}  📍 {loc_str}")
            print()

            a_fill  = int(r["team_a_prob"] * 20)
            b_fill  = int(r["team_b_prob"] * 20)
            a_pick  = "  ← PICK" if r["team_a_prob"] > r["team_b_prob"] else ""
            b_pick  = "  ← PICK" if r["team_b_prob"] > r["team_a_prob"] else ""

            print(f"  ({int(r['home_seed']):>2}) {r['home_team']:<30} "
                  f"{r['team_a_prob']*100:5.1f}%  "
                  f"{'█'*a_fill}{'░'*(20-a_fill)}{a_pick}")
            print(f"  ({int(r['away_seed']):>2}) {r['away_team']:<30} "
                  f"{r['team_b_prob']*100:5.1f}%  "
                  f"{'█'*b_fill}{'░'*(20-b_fill)}{b_pick}")
            print()

            if r["spread"] >= 0:
                spread_str = f"{r['home_team'].split()[-1]} -{abs(r['spread']):.1f}"
            else:
                spread_str = f"{r['away_team'].split()[-1]} -{abs(r['spread']):.1f}"

            print(f"  Spread: {spread_str:<25} Est. Total: {r['predicted_total']:.0f} pts")

    print()
    print("=" * 72)
    print("\n  SUMMARY")
    print(f"  {'Game':<44} {'Pick':<24} {'Prob':>6}  Conf")
    print(f"  {'─'*44} {'─'*24} {'─'*6}  {'─'*8}")

    for r in results:
        if r["team_a_prob"] >= r["team_b_prob"]:
            pick, prob = r["home_team"], r["team_a_prob"]
        else:
            pick, prob = r["away_team"], r["team_b_prob"]
        game = (f"({int(r['home_seed'])}) {r['home_team'].split()[-1]} vs "
                f"({int(r['away_seed'])}) {r['away_team'].split()[-1]}")
        icon = CONFIDENCE_ICONS.get(r["confidence"], "")
        print(f"  {game:<44} {pick:<24} {prob*100:5.1f}%  {icon} {r['confidence']}")

    locks   = sum(1 for r in results if r["confidence"] == "LOCK")
    leans   = sum(1 for r in results if r["confidence"] == "LEAN")
    tossups = sum(1 for r in results if r["confidence"] == "TOSS-UP")
    print(f"\n  {len(results)} games  |  🔒 {locks} Locks  📊 {leans} Leans  🪙 {tossups} Toss-Ups\n")


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def write_markdown(results: list[dict], output_path: Path) -> None:
    """
    Writes a GitHub-renderable markdown prediction report.
    Saved to docs/predictions_MMDD.md so it renders on GitHub automatically.
    """
    lines = []
    date_str = datetime.now().strftime("%B %d, %Y")
    time_str = datetime.now().strftime("%I:%M %p ET")

    lines.append(f"# NCAA Tournament Predictions — {date_str}\n")
    lines.append(f"*Generated by ML model &nbsp;|&nbsp; {len(results)} games "
                 f"&nbsp;|&nbsp; {time_str}*\n")
    lines.append("> **Model:** Gradient Boosting trained on 2018–2025 NCAAB seasons  \n"
                 "> **Method:** Monte Carlo bracket simulation, neutral court adjustment  \n"
                 "> **Tiers:** 🔒 Lock ≥75% · 📊 Lean 60–74% · 🪙 Toss-Up <60%\n")

    # Group by round
    by_round = {}
    for r in results:
        rnd = r["round"]
        if rnd not in by_round:
            by_round[rnd] = []
        by_round[rnd].append(r)

    for rnd in sorted(by_round.keys()):
        round_name = ROUND_NAMES.get(rnd, f"Round {rnd}")
        lines.append(f"\n## {round_name}\n")
        lines.append("| | Team | Seed | Win Prob | Spread | Est. Total | Confidence |")
        lines.append("|:---:|---|:---:|---:|---|---:|:---:|")

        for r in by_round[rnd]:
            icon = CONFIDENCE_ICONS.get(r["confidence"], "")

            if r["team_a_prob"] >= r["team_b_prob"]:
                h_marker, a_marker = "✅", ""
                prob_h = f"**{r['team_a_prob']*100:.1f}%**"
                prob_a = f"{r['team_b_prob']*100:.1f}%"
            else:
                h_marker, a_marker = "", "✅"
                prob_h = f"{r['team_a_prob']*100:.1f}%"
                prob_a = f"**{r['team_b_prob']*100:.1f}%**"

            if r["spread"] >= 0:
                spread_str = f"{r['home_team'].split()[-1]} -{abs(r['spread']):.1f}"
            else:
                spread_str = f"{r['away_team'].split()[-1]} -{abs(r['spread']):.1f}"

            total_str = f"{r['predicted_total']:.0f} pts"
            conf_str  = f"{icon} {r['confidence']}"

            dt      = r.get("date_parsed")
            dt_str  = dt.strftime("%a %b %d %I:%M %p") if pd.notna(dt) else ""
            loc_str = r.get("location") or ""

            # Home row
            lines.append(
                f"| {h_marker} | **{r['home_team']}** | {int(r['home_seed'])} | "
                f"{prob_h} | {spread_str} | {total_str} | {conf_str} |"
            )
            # Away row
            lines.append(
                f"| {a_marker} | {r['away_team']} | {int(r['away_seed'])} | "
                f"{prob_a} | | | |"
            )
            # Date/location row
            if dt_str or loc_str:
                info = " · ".join(filter(None, [dt_str, loc_str]))
                lines.append(f"| | *{info}* | | | | | |")

            # If game is completed, show result
            if r["status"] == "post" and pd.notna(r.get("home_score")):
                result_str = (f"Final: {r['home_team'].split()[-1]} "
                              f"{int(r['home_score'])} – "
                              f"{int(r['away_score'])} "
                              f"{r['away_team'].split()[-1]}")
                winner     = r["home_team"] if r.get("home_winner") else r["away_team"]
                correct    = "✅ Correct" if (
                    (r["team_a_prob"] >= r["team_b_prob"] and r.get("home_winner")) or
                    (r["team_b_prob"] >  r["team_a_prob"] and not r.get("home_winner"))
                ) else "❌ Incorrect"
                lines.append(f"| | *{result_str} — {correct}* | | | | | |")

            lines.append("|  |  |  |  |  |  |  |")  # spacer

    # Summary table
    lines.append("\n---\n")
    lines.append("## Summary\n")
    lines.append("| Game | Pick | Probability | Spread | Confidence |")
    lines.append("|---|---|:---:|---|:---:|")

    for r in results:
        if r["team_a_prob"] >= r["team_b_prob"]:
            pick, prob = r["home_team"], r["team_a_prob"]
        else:
            pick, prob = r["away_team"], r["team_b_prob"]

        game = (f"({int(r['home_seed'])}) {r['home_team'].split()[-1]} vs "
                f"({int(r['away_seed'])}) {r['away_team'].split()[-1]}")

        if r["spread"] >= 0:
            spread_str = f"{r['home_team'].split()[-1]} -{abs(r['spread']):.1f}"
        else:
            spread_str = f"{r['away_team'].split()[-1]} -{abs(r['spread']):.1f}"

        icon = CONFIDENCE_ICONS.get(r["confidence"], "")
        lines.append(
            f"| {game} | **{pick}** | {prob*100:.1f}% | "
            f"{spread_str} | {icon} {r['confidence']} |"
        )

    # Footer stats
    locks   = sum(1 for r in results if r["confidence"] == "LOCK")
    leans   = sum(1 for r in results if r["confidence"] == "LEAN")
    tossups = sum(1 for r in results if r["confidence"] == "TOSS-UP")

    lines.append(f"\n---\n")
    lines.append(
        f"*🔒 {locks} Locks &nbsp;·&nbsp; 📊 {leans} Leans "
        f"&nbsp;·&nbsp; 🪙 {tossups} Toss-Ups*  \n"
        f"*[View full bracket predictions](bracket_predictions.csv)*"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown saved -> %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict individual NCAA Tournament games"
    )
    parser.add_argument("--data-dir",  default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--season",    type=int, default=2025)
    parser.add_argument("--round",     type=int, default=None,
                        help="Filter to round (0=First Four, 1=R64, 2=R32, etc.)")
    parser.add_argument("--date",      default=None,
                        help="Filter to date YYYY-MM-DD")
    parser.add_argument("--today",     action="store_true",
                        help="Shortcut for --date today")
    parser.add_argument("--status",    default=None,
                        choices=["pre", "in", "post"],
                        help="Filter by status: pre / in / post")
    parser.add_argument("--no-save",   action="store_true",
                        help="Skip saving CSV and markdown")
    return parser.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # --today shortcut
    if args.today:
        args.date = datetime.now().strftime("%Y-%m-%d")
        log.info("--today: filtering to %s", args.date)

    # Load model
    model_path = model_dir / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Run train_model.py first.")
    model = joblib.load(model_path)
    log.info("Loaded model from %s", model_path)

    # Pull games
    games = pull_tournament_games(
        season        = args.season,
        round_filter  = args.round,
        date_filter   = args.date,
        status_filter = args.status,
    )

    if games.empty:
        print("\nNo games found matching filters.")
        print("Tips:")
        print("  • Try --today for today's games")
        print("  • Try removing --status or --round filters")
        print("  • Check that the bracket is populated for --season", args.season)
        return

    # Load features
    team_ids = list(set(
        games["home_team_id"].dropna().astype(int).tolist() +
        games["away_team_id"].dropna().astype(int).tolist()
    ))
    team_features = load_team_features(data_dir, args.season, team_ids)

    def get_features(tid: int) -> pd.Series | None:
        rows = team_features[team_features["team_id"] == tid]
        return rows.iloc[0] if len(rows) > 0 else None

    # Generate predictions
    results   = []
    n_skipped = 0

    for _, game in games.iterrows():
        h_id = int(game["home_team_id"]) if pd.notna(game["home_team_id"]) else None
        a_id = int(game["away_team_id"]) if pd.notna(game["away_team_id"]) else None

        if not h_id or not a_id:
            n_skipped += 1
            continue

        h_feat = get_features(h_id)
        a_feat = get_features(a_id)

        if h_feat is None or a_feat is None:
            log.warning("Missing features: %s vs %s — skipping",
                        game["home_team"], game["away_team"])
            n_skipped += 1
            continue

        pred = predict_game(model, h_feat, a_feat)

        results.append({
            "game_id":        game["game_id"],
            "round":          int(game["round"]) if pd.notna(game["round"]) else None,
            "round_name":     ROUND_NAMES.get(int(game["round"]) if pd.notna(game["round"]) else -1, ""),
            "date":           game["date"],
            "date_parsed":    game["date_parsed"],
            "location":       game["location"],
            "status":         game["status"],
            "status_detail":  game["status_detail"],
            "home_team_id":   h_id,
            "home_team":      game["home_team"],
            "home_seed":      game["home_seed"],
            "home_score":     game["home_score"],
            "home_winner":    game["home_winner"],
            "away_team_id":   a_id,
            "away_team":      game["away_team"],
            "away_seed":      game["away_seed"],
            "away_score":     game["away_score"],
            "away_winner":    game["away_winner"],
            **pred,
        })

    if n_skipped > 0:
        log.warning("Skipped %d games due to missing data", n_skipped)

    if not results:
        print("No predictions could be generated.")
        return

    # Print to terminal
    print_predictions(results)

    if not args.no_save:
        date_tag = datetime.now().strftime("%m%d")

        # CSV — dated filename, overwrite
        csv_path = model_dir / f"predictions_{date_tag}.csv"
        df_out   = pd.DataFrame(results)
        df_out["predicted_at"] = datetime.now().isoformat()
        df_out.to_csv(csv_path, index=False)
        log.info("CSV saved -> %s", csv_path)

        # Markdown — goes into docs/ for GitHub rendering
        md_path = Path("docs") / f"predictions_{date_tag}.md"
        write_markdown(results, md_path)

        print(f"  Files saved:")
        print(f"    {csv_path}")
        print(f"    {md_path}")
        print(f"\n  To publish predictions:")
        print(f"    git add docs/predictions_{date_tag}.md models/predictions_{date_tag}.csv")
        print(f"    git commit -m \"Predictions {datetime.now().strftime('%B %d')}")
        print(f"    git push\n")


if __name__ == "__main__":
    main()