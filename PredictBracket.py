"""
predict_bracket.py
------------------
Pulls the 2025 NCAA Tournament bracket from BallDontLie, joins current
season team features, and runs a Monte Carlo simulation to estimate
each team's probability of reaching every round.

Round structure per region:
  R64  (round 1): 16 teams, 8 games -> 8 winners enter R32
  R32  (round 2): 8 teams,  4 games -> 4 winners enter S16
  S16  (round 3): 4 teams,  2 games -> 2 winners enter E8
  E8   (round 4): 2 teams,  1 game  -> 1 regional champion
  F4   (round 5): 4 champs, 2 games -> 2 finalists
  Chmp (round 6): 2 teams,  1 game  -> 1 champion

Key design: R64 is simulated game-by-game producing 8 winners.
Those 8 winners are then paired sequentially for R32 (0 vs 1, 2 vs 3,
etc.) which correctly implements the bracket structure:
  Game 1 winner (1/16) vs Game 2 winner (8/9)
  Game 3 winner (5/12) vs Game 4 winner (4/13)
  Game 5 winner (6/11) vs Game 6 winner (3/14)
  Game 7 winner (7/10) vs Game 8 winner (2/15)

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
    "conference_wins", "conference_losses",
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


# ---------------------------------------------------------------------------
# Pull bracket — paginated
# ---------------------------------------------------------------------------

def pull_bracket(season: int) -> pd.DataFrame:
    log.info("Pulling bracket for season %d ...", season)

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
                wait = 2 ** attempt
                log.warning("Rate limited — sleeping %ds", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Bracket not found for season {season}. Try --season 2025."
                )
            resp.raise_for_status()
            break

        data = resp.json()
        records.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        log.info("  Paginating — %d records so far ...", len(records))
        time.sleep(RATE_LIMIT_SLEEP)

    log.info("Bracket API returned %d total records", len(records))
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

    r64_count = (df["round"] == 1).sum()
    log.info(
        "Bracket: %d total slots | rounds: %s | Round of 64 games: %d / 32",
        len(df),
        sorted(df["round"].dropna().unique().tolist()),
        r64_count,
    )
    if r64_count < 32:
        log.warning(
            "Only %d / 32 Round of 64 games — bracket may be incomplete.",
            r64_count,
        )

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

    log.info("Feature vectors: %d / %d bracket teams",
             len(tf_bracket), len(bracket_team_ids))
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
        conf_map.rename(columns={"team_id": "home_team_id",
                                  "conference_id": "home_conf_id"}),
        on=["home_team_id", "season"], how="left"
    )
    g = g.merge(
        conf_map.rename(columns={"team_id": "away_team_id",
                                  "conference_id": "away_conf_id"}),
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
        pd.concat([home_nc[["conference_id", "win"]],
                   away_nc[["conference_id", "win"]]])
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
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_game(
    a: int | None,
    b: int | None,
    rnd: int,
    round_counts: dict,
    win_prob_fn,
) -> int | None:
    """
    Simulates a single game. Increments winner's round count.
    Returns winning team_id.
    """
    if a is None and b is None:
        return None
    if a is None:
        w = b
    elif b is None:
        w = a
    else:
        p = win_prob_fn(a, b)
        w = a if np.random.random() < p else b

    if w and w in round_counts:
        round_counts[w][rnd] += 1

    return w


def simulate_rounds(
    current: list,
    n_rounds: int,
    round_start: int,
    round_counts: dict,
    win_prob_fn,
) -> list:
    """
    Simulates n_rounds of single-elimination starting from round_start.
    Pairs sequentially: index 0 vs 1, 2 vs 3, etc.
    Returns list of winners (half the size of input each round).
    """
    for rnd in range(round_start, round_start + n_rounds):
        if len(current) <= 1:
            break

        next_round = []
        i = 0
        while i < len(current):
            if i + 1 >= len(current):
                # Odd team out — bye
                bye = current[i]
                next_round.append(bye)
                if bye and bye in round_counts:
                    round_counts[bye][rnd] += 1
                break
            w = simulate_game(
                a            = current[i],
                b            = current[i + 1],
                rnd          = rnd,
                round_counts = round_counts,
                win_prob_fn  = win_prob_fn,
            )
            next_round.append(w)
            i += 2

        current = next_round

    return current


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
    Simulates the full tournament bracket.

    Region simulation (per region, per simulation):
      1. Simulate each R64 game individually -> 8 winners
      2. Pair those 8 winners for R32 (game 1 winner vs game 2 winner, etc.)
      3. simulate_rounds for R32 (round 2) -> 4 teams
      4. simulate_rounds for S16 (round 3) -> 2 teams
      5. simulate_game for E8 (round 4)    -> 1 regional champion

    This correctly implements standard bracket pairing at every round.
    """
    log.info("Running %d Monte Carlo simulations ...", n_simulations)

    playin_games = (
        bracket[bracket["round"] == 0]
        .copy()
        .sort_values("bracket_location")
        .reset_index(drop=True)
    )
    r64_games = (
        bracket[bracket["round"] == 1]
        .copy()
        .sort_values("bracket_location")
        .reset_index(drop=True)
    )

    if r64_games.empty:
        raise RuntimeError(
            "No Round of 64 games found (round=1). "
            f"Rounds present: {sorted(bracket['round'].unique().tolist())}"
        )

    # ------------------------------------------------------------------
    # Classify teams
    # ------------------------------------------------------------------
    r64_team_ids = set(
        r64_games["home_team_id"].dropna().astype(int).tolist() +
        r64_games["away_team_id"].dropna().astype(int).tolist()
    )

    active_playin = []
    for _, row in playin_games.iterrows():
        h_id = int(row["home_team_id"]) if pd.notna(row["home_team_id"]) else None
        a_id = int(row["away_team_id"]) if pd.notna(row["away_team_id"]) else None

        h_confirmed = h_id in r64_team_ids if h_id else False
        a_confirmed = a_id in r64_team_ids if a_id else False

        if h_confirmed or a_confirmed:
            log.info("  Play-in already resolved (winner in R64): %s vs %s",
                     row["home_team_name"], row["away_team_name"])
            continue

        active_playin.append({
            "home_id":          h_id,
            "away_id":          a_id,
            "home_name":        row["home_team_name"],
            "away_name":        row["away_team_name"],
            "seed":             int(row["home_seed"]) if pd.notna(row["home_seed"]) else None,
            "bracket_location": row["bracket_location"],
        })

    log.info("Active play-in games: %d", len(active_playin))

    # ------------------------------------------------------------------
    # Map play-in games to TBD slots by bracket_location order
    # ------------------------------------------------------------------
    tbd_slots = []
    for _, row in r64_games.iterrows():
        if pd.isna(row["home_team_id"]) or pd.isna(row["away_team_id"]):
            tbd_slots.append(int(row["bracket_location"]))

    active_playin.sort(key=lambda x: x["bracket_location"])
    tbd_slots.sort()

    tbd_to_playin_idx: dict[int, int] = {}
    for i, loc in enumerate(tbd_slots):
        if i < len(active_playin):
            tbd_to_playin_idx[loc] = i

    if tbd_to_playin_idx:
        log.info("TBD slot mappings:")
        for loc, idx in tbd_to_playin_idx.items():
            g = active_playin[idx]
            log.info("  bracket_location=%d -> %s vs %s",
                     loc, g["home_name"], g["away_name"])

    # ------------------------------------------------------------------
    # Derive regions (every 8 bracket_location slots = one region)
    # ------------------------------------------------------------------
    r64_games["derived_region"] = (r64_games["bracket_location"] - 1) // 8
    regions = sorted(r64_games["derived_region"].unique().tolist())
    region_labels = {r: f"Region {r + 1}" for r in regions}

    log.info(
        "Derived %d regions | games per region: %s",
        len(regions),
        r64_games.groupby("derived_region").size().to_dict(),
    )

    region_bracket: dict[int, list[dict]] = {}
    for region_id in regions:
        rg = r64_games[
            r64_games["derived_region"] == region_id
        ].sort_values("bracket_location")
        matchups = []
        for _, row in rg.iterrows():
            matchups.append({
                "home_id":          int(row["home_team_id"]) if pd.notna(row["home_team_id"]) else None,
                "away_id":          int(row["away_team_id"]) if pd.notna(row["away_team_id"]) else None,
                "bracket_location": int(row["bracket_location"]),
            })
        region_bracket[region_id] = matchups

    # ------------------------------------------------------------------
    # Build team info
    # ------------------------------------------------------------------
    team_info: dict[int, dict] = {}

    for _, row in r64_games.iterrows():
        region_num = int(row["derived_region"])
        for tid, name, seed in [
            (row["home_team_id"], row["home_team_name"], row["home_seed"]),
            (row["away_team_id"], row["away_team_name"], row["away_seed"]),
        ]:
            if pd.notna(tid):
                tid = int(tid)
                if tid not in team_info:
                    team_info[tid] = {
                        "team_name":   name,
                        "seed":        seed,
                        "region":      region_labels[region_num],
                        "playin_only": False,
                    }

    for game in active_playin:
        for tid, name in [
            (game["home_id"], game["home_name"]),
            (game["away_id"], game["away_name"]),
        ]:
            if tid and tid not in team_info:
                team_info[tid] = {
                    "team_name":   name,
                    "seed":        game["seed"],
                    "region":      "First Four",
                    "playin_only": True,
                }

    all_team_ids = list(team_info.keys())
    log.info(
        "Teams tracked: %d total (%d play-in only, %d confirmed R64)",
        len(all_team_ids),
        sum(1 for t in team_info.values() if t["playin_only"]),
        sum(1 for t in team_info.values() if not t["playin_only"]),
    )

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

    def win_prob_fn(a, b):
        if a is None or b is None:
            return 0.5
        return prob_cache.get((a, b), 0.5)

    # ------------------------------------------------------------------
    # Round counts
    # round 0 = play-in appearance
    # rounds 1-6 = R64 through Champion
    # ------------------------------------------------------------------
    round_counts = {tid: {r: 0 for r in range(0, 8)} for tid in all_team_ids}

    for tid, info in team_info.items():
        if info["playin_only"]:
            round_counts[tid][0] = n_simulations

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    for sim in range(n_simulations):
        if sim % 2000 == 0 and sim > 0:
            log.info("  Simulation %d / %d ...", sim, n_simulations)

        # Step 1: Simulate play-in games
        playin_results: list[int | None] = []
        for game in active_playin:
            h, a = game["home_id"], game["away_id"]
            if h is None and a is None:
                playin_results.append(None)
                continue
            if h is None:
                w = a
            elif a is None:
                w = h
            else:
                p = win_prob_fn(h, a)
                w = h if np.random.random() < p else a
            playin_results.append(w)
            # Play-in winner counts as entering R64
            if w and w in round_counts:
                round_counts[w][1] += 1

        # Step 2: Simulate each region -> 1 regional champion
        region_champions: list[int] = []

        for region_id, matchups in region_bracket.items():

            # --- R64: simulate each game, collect 8 winners ---
            r64_winners: list[int | None] = []

            for m in matchups:
                h   = m["home_id"]
                a   = m["away_id"]
                loc = m["bracket_location"]

                # Substitute TBD slots with play-in winners
                pi_idx = tbd_to_playin_idx.get(loc)
                if h is None and pi_idx is not None and pi_idx < len(playin_results):
                    h = playin_results[pi_idx]
                if a is None and pi_idx is not None and pi_idx < len(playin_results):
                    a = playin_results[pi_idx]

                # Count R64 appearances for confirmed teams
                for tid in [h, a]:
                    if (tid and tid in round_counts
                            and not team_info.get(tid, {}).get("playin_only")):
                        round_counts[tid][1] += 1

                # Simulate R64 game — winner enters R32 (round 2)
                w = simulate_game(
                    a            = h,
                    b            = a,
                    rnd          = 2,
                    round_counts = round_counts,
                    win_prob_fn  = win_prob_fn,
                )
                r64_winners.append(w)

            # r64_winners is now [w1, w2, w3, w4, w5, w6, w7, w8]
            # Pairing: w1 vs w2, w3 vs w4, w5 vs w6, w7 vs w8
            # which gives: (1/16 winner) vs (8/9 winner), etc.
            current = r64_winners

            # --- S16: simulate R32 winners ---
            current = simulate_rounds(
                current      = current,
                n_rounds     = 1,
                round_start  = 3,
                round_counts = round_counts,
                win_prob_fn  = win_prob_fn,
            )

            # --- E8: simulate S16 winners ---
            current = simulate_rounds(
                current      = current,
                n_rounds     = 1,
                round_start  = 4,
                round_counts = round_counts,
                win_prob_fn  = win_prob_fn,
            )

            # --- Regional final: simulate E8 winners -> 1 champion ---
            if len(current) >= 2:
                champion = simulate_game(
                    a            = current[0],
                    b            = current[1],
                    rnd          = 5,
                    round_counts = round_counts,
                    win_prob_fn  = win_prob_fn,
                )
            elif len(current) == 1:
                champion = current[0]
                if champion and champion in round_counts:
                    round_counts[champion][5] += 1
            else:
                champion = None

            if champion is not None:
                region_champions.append(champion)

        # Step 3: Final Four semifinal (round 6) -> 2 finalists
        if len(region_champions) < 2:
            continue

        ff_winners = simulate_rounds(
            current      = region_champions,
            n_rounds     = 1,
            round_start  = 6,
            round_counts = round_counts,
            win_prob_fn  = win_prob_fn,
        )

        # Step 4: Championship (round 7 internally, shown as Champion)
        if len(ff_winners) >= 2:
            simulate_game(
                a            = ff_winners[0],
                b            = ff_winners[1],
                rnd          = 7,
                round_counts = round_counts,
                win_prob_fn  = win_prob_fn,
            )
        elif len(ff_winners) == 1 and ff_winners[0] is not None:
            if ff_winners[0] in round_counts:
                round_counts[ff_winners[0]][7] += 1

    # ------------------------------------------------------------------
    # Convert counts to probabilities
    # ------------------------------------------------------------------
    results = []
    for tid, info in team_info.items():
        results.append({
            "team_id":          tid,
            "team_name":        info["team_name"],
            "seed":             info["seed"],
            "region":           info["region"],
            # round 1 = R64 appearance
            # round 2 = won R64 (entered R32)
            # round 3 = won R32 (entered S16)
            # round 4 = won S16 (entered E8)
            # round 5 = won E8 (regional champ / entered F4)
            # round 6 = won F4 semifinal (entered championship)
            # round 7 = won championship
            "Round of 64_prob": round_counts[tid][1] / n_simulations,
            "Round of 32_prob": round_counts[tid][2] / n_simulations,
            "Sweet 16_prob":    round_counts[tid][3] / n_simulations,
            "Elite 8_prob":     round_counts[tid][4] / n_simulations,
            "Final Four_prob":  round_counts[tid][5] / n_simulations,
            "Champion_prob":    round_counts[tid][7] / n_simulations,
        })

    df = pd.DataFrame(results)
    df = df.sort_values(["region", "seed"]).reset_index(drop=True)
    log.info("Simulation complete.")
    return df


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(predictions: pd.DataFrame) -> None:
    """
    Champion probs should sum to ~1.0 (one champion per simulation).
    Final Four probs should sum to ~4.0 (four teams reach F4 each sim).
    R64 probs should sum to ~64.0 (all 64 teams play R64).
    """
    champ_sum = predictions["Champion_prob"].sum()
    ff_sum    = predictions["Final Four_prob"].sum()
    r64_sum   = predictions["Round of 64_prob"].sum()

    log.info("Sanity check:")
    log.info("  Champion probs sum:    %.3f (expect ~1.0)", champ_sum)
    log.info("  Final Four probs sum:  %.3f (expect ~4.0)", ff_sum)
    log.info("  Round of 64 probs sum: %.3f (expect ~64.0)", r64_sum)

    if abs(champ_sum - 1.0) > 0.05:
        log.warning("  Champion probs sum is off — check simulation logic.")
    if abs(ff_sum - 4.0) > 0.2:
        log.warning("  Final Four probs sum is off — check simulation logic.")
    if abs(r64_sum - 64.0) > 1.0:
        log.warning("  Round of 64 probs sum is off — check simulation logic.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(predictions: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("=" * 95)
    lines.append("MARCH MADNESS 2025 -- MONTE CARLO BRACKET PREDICTIONS")
    lines.append("=" * 95)

    round_cols = [
        "Round of 64_prob", "Round of 32_prob", "Sweet 16_prob",
        "Elite 8_prob", "Final Four_prob", "Champion_prob",
    ]
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
            ff    = row.get(ff_col,    0) or 0
            e8    = row.get(e8_col,    0) or 0
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
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--model-dir",   default="models")
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--season",      type=int, default=2025)
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

    sanity_check(predictions)
    generate_report(predictions, model_dir / "bracket_report.txt")


if __name__ == "__main__":
    main()