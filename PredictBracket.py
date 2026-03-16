"""
predict_bracket.py
------------------
Pulls the 2025 NCAA Tournament bracket from BallDontLie, joins current
season team features, and runs a Monte Carlo simulation to estimate
each team's probability of reaching every round.

API bracket round numbering:
  0 = First Four / play-in games
  1 = Round of 64 (all first-round games)
  2-6 don't exist yet in the API — simulated internally

Notes on 2025 bracket:
  - region_id and region_label are null — regions derived from bracket_location
  - Bracket may be incomplete (some regions missing games) — handled gracefully
  - Every 8 bracket_location slots = one region
  - Final Four handles 2, 3, or 4 regional champions

Usage:
  python predict_bracket.py [--data-dir data] [--model-dir models]
                            [--simulations 10000] [--season 2025]
"""

import os
import time
import argparse
import logging
from pathlib import Path

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

ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Champion",
}

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


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_headers() -> dict:
    api_key = os.environ.get("BDL_API_KEY")
    if not api_key:
        raise EnvironmentError("Set BDL_API_KEY environment variable.")
    return {"Authorization": api_key}


def api_get(endpoint: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(5):
        resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
        if resp.status_code == 429:
            wait = 2 ** attempt
            log.warning("Rate limited — sleeping %ds", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed after retries: {url}")


# ---------------------------------------------------------------------------
# Pull bracket
# ---------------------------------------------------------------------------

def pull_bracket(season: int) -> pd.DataFrame:
    log.info("Pulling bracket for season %d ...", season)
    try:
        data = api_get("bracket", {"season": season})
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Bracket not found for season {season}. Try --season 2025."
            ) from e
        raise

    records = data.get("data", [])
    if not records:
        raise RuntimeError(f"Bracket returned empty for season {season}.")

    rows = []
    for entry in records:
        home = entry.get("home_team") or {}
        away = entry.get("away_team") or {}
        rows.append({
            "game_id":          entry.get("game_id"),
            "season":           entry.get("season"),
            "round":            entry.get("round"),
            "region_id":        entry.get("region_id"),
            "region_label":     entry.get("region_label"),
            "bracket_location": entry.get("bracket_location"),
            "date":             entry.get("date"),
            "location":         entry.get("location"),
            "status":           entry.get("status"),
            "home_team_id":     home.get("id"),
            "home_team_name":   home.get("full_name") or home.get("name"),
            "home_seed":        home.get("seed"),
            "home_score":       home.get("score"),
            "home_winner":      home.get("winner"),
            "away_team_id":     away.get("id"),
            "away_team_name":   away.get("full_name") or away.get("name"),
            "away_seed":        away.get("seed"),
            "away_score":       away.get("score"),
            "away_winner":      away.get("winner"),
        })

    df = pd.DataFrame(rows)
    df["home_seed"] = pd.to_numeric(df["home_seed"], errors="coerce")
    df["away_seed"] = pd.to_numeric(df["away_seed"], errors="coerce")

    log.info("Bracket: %d game slots | rounds present: %s",
             len(df), sorted(df["round"].unique().tolist()))
    return df


# ---------------------------------------------------------------------------
# Build team features
# ---------------------------------------------------------------------------

def build_tournament_features(
    bracket: pd.DataFrame,
    data_dir: Path,
    season: int,
) -> pd.DataFrame:
    log.info("Building tournament team features for season %d ...", season)

    tf_path  = data_dir / "team_features.parquet"
    raw_path = data_dir / "games.parquet"

    if not tf_path.exists():
        raise FileNotFoundError(f"team_features.parquet not found in {data_dir}")

    tf        = pd.read_parquet(tf_path)
    raw_games = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()

    tf_season = tf[tf["season"] == season].copy()
    if tf_season.empty:
        raise RuntimeError(
            f"No team features found for season {season}. "
            f"Re-run collect_data.py with --seasons {season}."
        )

    if not raw_games.empty:
        tf_season = _add_conference_features(raw_games, tf_season, season)

    bracket_team_ids = set(
        bracket["home_team_id"].dropna().astype(int).tolist() +
        bracket["away_team_id"].dropna().astype(int).tolist()
    )

    tf_bracket = tf_season[tf_season["team_id"].isin(bracket_team_ids)].copy()
    missing    = bracket_team_ids - set(tf_bracket["team_id"].tolist())
    if missing:
        log.warning("%d bracket teams missing from features: %s", len(missing), missing)

    log.info("Feature vectors: %d / %d bracket teams", len(tf_bracket), len(bracket_team_ids))
    return tf_bracket


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
        conf_games.groupby("home_conf_id")["total_score"].mean()
        .reset_index()
        .rename(columns={"home_conf_id": "conference_id", "total_score": "conf_pace"})
    )
    depth = (
        conf_games.groupby("home_conf_id")["abs_margin"].mean()
        .reset_index()
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
        .groupby("conference_id")["win"].mean()
        .reset_index()
        .rename(columns={"win": "conf_strength"})
    )

    conf_stats = (
        pace
        .merge(strength, on="conference_id", how="outer")
        .merge(depth,    on="conference_id", how="outer")
    )
    return tf.merge(conf_stats, on="conference_id", how="left")


# ---------------------------------------------------------------------------
# Win probability
# ---------------------------------------------------------------------------

def build_matchup_features(
    team_a_id: int,
    team_b_id: int,
    team_features: pd.DataFrame,
) -> pd.DataFrame | None:
    all_features = BASE_FEATURES + CONF_BASE_FEATURES

    def get_team(tid):
        row = team_features[team_features["team_id"] == tid]
        return row.iloc[0] if len(row) > 0 else None

    a = get_team(team_a_id)
    b = get_team(team_b_id)
    if a is None or b is None:
        return None

    row = {}
    for f in all_features:
        ha = a.get(f, np.nan) if hasattr(a, "get") else getattr(a, f, np.nan)
        hb = b.get(f, np.nan) if hasattr(b, "get") else getattr(b, f, np.nan)
        row[f"home_{f}"] = ha
        row[f"away_{f}"] = hb
        try:
            row[f"diff_{f}"] = float(ha) - float(hb)
        except (TypeError, ValueError):
            row[f"diff_{f}"] = np.nan

    return pd.DataFrame([row])


def predict_win_prob(
    model,
    team_a_id: int,
    team_b_id: int,
    team_features: pd.DataFrame,
) -> float:
    """P(team_a beats team_b) on neutral court — averages both orientations."""
    feat_ab = build_matchup_features(team_a_id, team_b_id, team_features)
    feat_ba = build_matchup_features(team_b_id, team_a_id, team_features)

    if feat_ab is None or feat_ba is None:
        log.warning("Missing features for %d vs %d -- using 0.5", team_a_id, team_b_id)
        return 0.5

    try:
        model_features = model.named_steps["pre"].named_steps["imputer"].feature_names_in_
        feat_ab = feat_ab.reindex(columns=model_features)
        feat_ba = feat_ba.reindex(columns=model_features)
    except AttributeError:
        pass

    spread_ab = model.predict(feat_ab)[0]
    spread_ba = model.predict(feat_ba)[0]
    prob_ab   = expit( spread_ab / SPREAD_SCALE)
    prob_ba   = expit(-spread_ba / SPREAD_SCALE)
    return float((prob_ab + prob_ba) / 2)


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_bracket(
    bracket: pd.DataFrame,
    team_features: pd.DataFrame,
    model,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    Simulates the full tournament using the actual bracket structure.

    Region derivation:
      region_id is null in 2025 API so we derive regions from bracket_location.
      Every 8 consecutive slots = one region (standard bracket structure).
      Incomplete regions (< 8 games) are simulated with available games only.

    Final Four:
      Handles 2, 3, or 4 regional champions gracefully.
      If 3 champions: first two play a semifinal, third gets a bye.
      If 2 champions: play directly for championship.
    """
    log.info("Running %d Monte Carlo simulations ...", n_simulations)

    playin_games = bracket[bracket["round"] == 0].copy().reset_index(drop=True)
    r64_games    = bracket[bracket["round"] == 1].copy()

    if r64_games.empty:
        raise RuntimeError(
            "No Round of 64 games found (round=1). "
            f"Rounds present: {sorted(bracket['round'].unique().tolist())}"
        )

    # ------------------------------------------------------------------
    # Derive regions from bracket_location
    # Every 8 slots = one region; incomplete final region is kept as-is
    # ------------------------------------------------------------------
    r64_games = r64_games.sort_values("bracket_location").reset_index(drop=True)
    r64_games["derived_region"] = (r64_games["bracket_location"] - 1) // 8

    regions = sorted(r64_games["derived_region"].unique().tolist())
    log.info(
        "Derived %d regions from bracket_location: %s",
        len(regions),
        r64_games.groupby("derived_region").size().to_dict()
    )

    region_bracket: dict[int, list] = {}
    region_labels:  dict[int, str]  = {}

    for region_id in regions:
        rg = r64_games[r64_games["derived_region"] == region_id].sort_values("bracket_location")
        matchups = []
        for _, row in rg.iterrows():
            h = int(row["home_team_id"]) if pd.notna(row["home_team_id"]) else None
            a = int(row["away_team_id"]) if pd.notna(row["away_team_id"]) else None
            matchups.append((h, a, row["home_seed"], row["away_seed"]))
        region_bracket[region_id] = matchups
        region_labels[region_id]  = f"Region {region_id + 1}"

    # ------------------------------------------------------------------
    # Play-in lookup: keyed by (row_index, seed) to keep games distinct
    # ------------------------------------------------------------------
    playin_lookup: dict[tuple, list[int]] = {}
    for idx, row in playin_games.iterrows():
        key = (idx, int(row["home_seed"]))
        participants = []
        if pd.notna(row["home_team_id"]):
            participants.append(int(row["home_team_id"]))
        if pd.notna(row["away_team_id"]):
            participants.append(int(row["away_team_id"]))
        if participants:
            playin_lookup[key] = participants

    playin_team_ids: set[int] = set()
    for participants in playin_lookup.values():
        playin_team_ids.update(participants)

    # ------------------------------------------------------------------
    # Build team info for output
    # ------------------------------------------------------------------
    team_info: dict[int, dict] = {}
    for _, row in pd.concat([r64_games, playin_games]).iterrows():
        for tid, name, seed, derived_region in [
            (row["home_team_id"], row["home_team_name"],
             row["home_seed"], row.get("derived_region")),
            (row["away_team_id"], row["away_team_name"],
             row["away_seed"], row.get("derived_region")),
        ]:
            if pd.notna(tid):
                tid = int(tid)
                if tid not in team_info:
                    region_num = int(derived_region) if pd.notna(derived_region) else None
                    team_info[tid] = {
                        "team_name": name,
                        "seed":      seed,
                        "region":    region_labels.get(region_num, "Play-In"),
                    }

    all_team_ids = list(team_info.keys())

    # ------------------------------------------------------------------
    # Precompute win probability cache
    # ------------------------------------------------------------------
    log.info("Precomputing win probabilities for %d teams ...", len(all_team_ids))
    prob_cache: dict[tuple, float] = {}
    for i, a in enumerate(all_team_ids):
        for b in all_team_ids[i + 1:]:
            p = predict_win_prob(model, a, b, team_features)
            prob_cache[(a, b)] = p
            prob_cache[(b, a)] = 1.0 - p

    def win_prob(a, b):
        if a is None or b is None:
            return 0.5
        return prob_cache.get((a, b), 0.5)

    def sample_winner(a, b):
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        p = win_prob(a, b)
        return a if np.random.random() < p else b

    # round_counts[team_id][round] where 1=R64 ... 6=Champion
    round_counts = {tid: {r: 0 for r in range(1, 7)} for tid in all_team_ids}

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    for sim in range(n_simulations):
        if sim % 2000 == 0 and sim > 0:
            log.info("  Simulation %d / %d ...", sim, n_simulations)

        # Step 1: resolve play-in games
        playin_winners: dict[tuple, int] = {}
        for key, participants in playin_lookup.items():
            if len(participants) >= 2:
                w = sample_winner(participants[0], participants[1])
            else:
                w = participants[0]
            playin_winners[key] = w

        # seed -> list of winners this sim (may be multiple per seed e.g. two 11-seed games)
        seed_winners_this_sim: dict[int, list[int]] = {}
        for key, winner in playin_winners.items():
            seed = key[1]
            if seed not in seed_winners_this_sim:
                seed_winners_this_sim[seed] = []
            seed_winners_this_sim[seed].append(winner)

        # Counter to assign play-in winners to slots in order per region
        seed_winner_cursor: dict[int, int] = {}

        def resolve(tid, seed):
            seed_int = int(seed) if pd.notna(seed) else None

            if tid is None:
                # TBD slot — assign next play-in winner for this seed
                if seed_int and seed_int in seed_winners_this_sim:
                    idx = seed_winner_cursor.get(seed_int, 0)
                    winners = seed_winners_this_sim[seed_int]
                    w = winners[idx % len(winners)]
                    seed_winner_cursor[seed_int] = idx + 1
                    return w
                return None

            tid = int(tid)
            if tid in playin_team_ids:
                # Find this team's play-in game and return that winner
                for key, participants in playin_lookup.items():
                    if tid in participants:
                        return playin_winners.get(key, tid)
                return tid

            return tid

        # Step 2: simulate each region through Elite 8
        region_champions: list[int] = []

        for region_id, matchups in region_bracket.items():
            # Reset cursor per region so each region gets its own play-in winners
            seed_winner_cursor.clear()

            # Build R64 field
            current: list[int | None] = []
            for h, a, h_seed, a_seed in matchups:
                rh = resolve(h, h_seed)
                ra = resolve(a, a_seed)
                current.append(rh)
                current.append(ra)
                if rh and rh in round_counts:
                    round_counts[rh][1] += 1
                if ra and ra in round_counts:
                    round_counts[ra][1] += 1

            # Simulate R32, Sweet 16, Elite 8 within this region
            # Stop when we have one team left (regional champion)
            for rnd in range(2, 5):
                if len(current) <= 1:
                    break
                next_round = []
                for i in range(0, len(current) - 1, 2):
                    w = sample_winner(current[i], current[i + 1])
                    next_round.append(w)
                    if w and w in round_counts:
                        round_counts[w][rnd] += 1
                # If odd team out (incomplete region), give them a bye
                if len(current) % 2 == 1:
                    bye_team = current[-1]
                    next_round.append(bye_team)
                    if bye_team and bye_team in round_counts:
                        round_counts[bye_team][rnd] += 1
                current = next_round

            if current:
                region_champions.append(current[0])

        # Step 3: Final Four
        # Handle 2, 3, or 4 regional champions
        ff_winners: list[int] = []

        if len(region_champions) == 4:
            # Standard: two semifinals
            for i in range(0, 4, 2):
                w = sample_winner(region_champions[i], region_champions[i + 1])
                ff_winners.append(w)
                if w and w in round_counts:
                    round_counts[w][5] += 1

        elif len(region_champions) == 3:
            # Two complete regions play a semifinal; third gets a bye to championship
            w1 = sample_winner(region_champions[0], region_champions[1])
            if w1 and w1 in round_counts:
                round_counts[w1][5] += 1
            w2 = region_champions[2]
            if w2 and w2 in round_counts:
                round_counts[w2][5] += 1
            ff_winners = [w1, w2]

        elif len(region_champions) == 2:
            # Both go straight to championship, both count as Final Four
            for w in region_champions:
                if w and w in round_counts:
                    round_counts[w][5] += 1
            ff_winners = region_champions

        elif len(region_champions) == 1:
            # Only one champion — they win by default
            champ = region_champions[0]
            if champ and champ in round_counts:
                round_counts[champ][5] += 1
                round_counts[champ][6] += 1
            continue

        # Step 4: Championship
        if len(ff_winners) >= 2:
            champ = sample_winner(ff_winners[0], ff_winners[1])
            if champ and champ in round_counts:
                round_counts[champ][6] += 1
        elif len(ff_winners) == 1:
            champ = ff_winners[0]
            if champ and champ in round_counts:
                round_counts[champ][6] += 1

    # ------------------------------------------------------------------
    # Convert counts to probabilities
    # ------------------------------------------------------------------
    results = []
    for tid, info in team_info.items():
        row = {
            "team_id":   tid,
            "team_name": info["team_name"],
            "seed":      info["seed"],
            "region":    info["region"],
        }
        for rnd, name in ROUND_NAMES.items():
            row[f"{name}_prob"] = round_counts[tid][rnd] / n_simulations
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values(["region", "seed"]).reset_index(drop=True)
    log.info("Simulation complete.")
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(predictions: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("=" * 95)
    lines.append("MARCH MADNESS 2025 -- MONTE CARLO BRACKET PREDICTIONS")
    lines.append("=" * 95)

    round_cols = [f"{ROUND_NAMES[r]}_prob" for r in range(1, 7)]
    round_cols = [c for c in round_cols if c in predictions.columns]

    header = f"{'Team':<35} {'Seed':>4}  " + "  ".join(
        f"{c.replace('_prob', ''):>12}" for c in round_cols
    )
    lines.append(header)
    lines.append("-" * len(header))

    for region in sorted(predictions["region"].dropna().unique()):
        region_df = predictions[predictions["region"] == region].sort_values("seed")
        lines.append(f"\n  -- {str(region).upper()} --")
        for _, row in region_df.iterrows():
            probs = "  ".join(
                f"{row[c]:>12.1%}" if pd.notna(row.get(c)) else f"{'N/A':>12}"
                for c in round_cols
            )
            seed = int(row["seed"]) if pd.notna(row["seed"]) else "?"
            lines.append(f"  {str(row['team_name']):<33} {seed:>4}  {probs}")

    lines.append("\n" + "=" * 95)
    lines.append("TOP 10 CHAMPIONSHIP CONTENDERS")
    lines.append("=" * 95)

    champ_col = "Champion_prob"
    ff_col    = "Final Four_prob"
    e8_col    = "Elite 8_prob"

    if champ_col in predictions.columns:
        top10 = predictions.nlargest(10, champ_col)
        for _, row in top10.iterrows():
            seed  = int(row["seed"]) if pd.notna(row["seed"]) else "?"
            champ = row.get(champ_col, 0) or 0
            ff    = row.get(ff_col, 0) or 0
            e8    = row.get(e8_col, 0) or 0
            lines.append(
                f"  {str(row['team_name']):<30} "
                f"({row['region']} {seed}-seed)  "
                f"Champion: {champ:.1%}  "
                f"Final Four: {ff:.1%}  "
                f"Elite 8: {e8:.1%}"
            )

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    log.info("Report saved -> %s", output_path)
    print("\n" + report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="March Madness bracket prediction")
    parser.add_argument("--data-dir",    default="data",   help="Directory with parquet files")
    parser.add_argument("--model-dir",   default="models", help="Directory with best_model.joblib")
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--season",      type=int, default=2025,
                        help="Season year (2025 = 2025 tournament)")
    return parser.parse_args()


def main():
    args      = parse_args()
    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    model_path = model_dir / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Run train_model.py first.")

    model = joblib.load(model_path)
    log.info("Loaded model from %s", model_path)

    bracket = pull_bracket(args.season)
    bracket.to_parquet(model_dir / "bracket_raw.parquet", index=False)

    team_features = build_tournament_features(bracket, data_dir, args.season)

    predictions = simulate_bracket(
        bracket, team_features, model, n_simulations=args.simulations
    )

    predictions.to_csv(model_dir / "bracket_predictions.csv", index=False)
    log.info("Predictions saved -> %s/bracket_predictions.csv", args.model_dir)

    generate_report(predictions, model_dir / "bracket_report.txt")


if __name__ == "__main__":
    main()