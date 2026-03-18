"""
pull_bracket.py
---------------
Pulls the current NCAA Tournament bracket from the BallDontLie API
and saves it to models/bracket_raw.parquet.

Uses the same pull_bracket() function as predict_bracket.py so there
is a single source of truth for bracket data. Validates completeness
and warns clearly if the bracket is not yet fully populated.

Run this independently before predict_bracket.py to refresh bracket
data as the API fills in missing games.

Usage:
  python pull_bracket.py [--season 2025] [--model-dir models]
"""

import os
import time
import argparse
import logging
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL             = "https://api.balldontlie.io/ncaab/v1"
EXPECTED_R64_GAMES   = 32
EXPECTED_REGIONS     = 4
GAMES_PER_REGION     = 8
EXPECTED_PLAYIN_GAMES = 4


# ---------------------------------------------------------------------------
# Shared bracket pull function (identical to predict_bracket.py)
# ---------------------------------------------------------------------------

def get_headers() -> dict:
    api_key = os.environ.get("BDL_API_KEY")
    if not api_key:
        raise EnvironmentError("Set BDL_API_KEY environment variable.")
    return {"Authorization": api_key}


def pull_bracket(season: int) -> pd.DataFrame:
    """
    Pulls the full tournament bracket from BallDontLie with pagination.
    Returns a flat DataFrame with one row per bracket game slot.

    This function is shared between pull_bracket.py and predict_bracket.py
    — any changes here should be reflected in both scripts.
    """
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
                    f"Bracket not found for season {season}. "
                    "The bracket may not be released yet. Try --season 2025."
                )
            resp.raise_for_status()
            break

        data = resp.json()
        records.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        log.info("  Paginating — %d records so far ...", len(records))
        time.sleep(0.12)

    log.info("Bracket API returned %d total records", len(records))

    if not records:
        raise RuntimeError(f"Bracket endpoint returned empty data for season {season}.")

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

    log.info(
        "Bracket: %d total slots | rounds present: %s",
        len(df), sorted(df["round"].dropna().unique().tolist())
    )
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_bracket(df: pd.DataFrame) -> bool:
    """
    Validates bracket completeness. Returns True if fully populated,
    False if incomplete. Logs detailed warnings for any missing pieces.
    """
    log.info("=" * 55)
    log.info("BRACKET VALIDATION")
    log.info("=" * 55)

    all_good = True

    # Play-in games
    playin = df[df["round"] == 0]
    log.info("First Four (round 0):   %d / %d expected",
             len(playin), EXPECTED_PLAYIN_GAMES)
    if len(playin) < EXPECTED_PLAYIN_GAMES:
        log.warning("  INCOMPLETE: Only %d of %d First Four games present.",
                    len(playin), EXPECTED_PLAYIN_GAMES)
        all_good = False

    # Round of 64
    r64 = df[df["round"] == 1].copy()
    log.info("Round of 64 (round 1):  %d / %d expected",
             len(r64), EXPECTED_R64_GAMES)
    if len(r64) < EXPECTED_R64_GAMES:
        log.warning("  INCOMPLETE: %d of %d Round of 64 games missing.",
                    EXPECTED_R64_GAMES - len(r64), EXPECTED_R64_GAMES)
        all_good = False

    # Region breakdown
    r64_sorted = r64.sort_values("bracket_location").reset_index(drop=True)
    r64_sorted["derived_region"] = (r64_sorted["bracket_location"] - 1) // GAMES_PER_REGION
    region_counts = r64_sorted.groupby("derived_region").size()

    log.info("Games per derived region:")
    for region_id, count in region_counts.items():
        status = "OK" if count == GAMES_PER_REGION else f"INCOMPLETE — {GAMES_PER_REGION - count} missing"
        log.info("  Region %d: %d games  [%s]", region_id + 1, count, status)
        if count < GAMES_PER_REGION:
            all_good = False

    n_regions = len(region_counts)
    if n_regions < EXPECTED_REGIONS:
        log.warning("  INCOMPLETE: Only %d of %d regions populated.",
                    n_regions, EXPECTED_REGIONS)
        all_good = False

    # TBD slots
    tbd_away = r64["away_team_id"].isna().sum()
    tbd_home = r64["home_team_id"].isna().sum()
    if tbd_home > 0 or tbd_away > 0:
        log.warning("  TBD slots: %d home, %d away — "
                    "will be filled by First Four winners in simulation.",
                    tbd_home, tbd_away)

    # Seed coverage
    all_seeds     = set(r64["home_seed"].dropna().astype(int).tolist() +
                        r64["away_seed"].dropna().astype(int).tolist())
    missing_seeds = set(range(1, 17)) - all_seeds
    if missing_seeds:
        log.warning("  Missing seed matchups: %s", sorted(missing_seeds))
        all_good = False

    log.info("=" * 55)
    if all_good:
        log.info("BRACKET STATUS: COMPLETE — ready for simulation.")
    else:
        log.warning(
            "BRACKET STATUS: INCOMPLETE — simulation will run with "
            "available data but probabilities may be skewed. "
            "Re-run once the API is fully populated."
        )
    log.info("=" * 55)

    return all_good


# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------

def print_bracket_summary(df: pd.DataFrame) -> None:
    r64 = df[df["round"] == 1].copy().sort_values("bracket_location")
    r64["derived_region"] = (r64["bracket_location"] - 1) // GAMES_PER_REGION

    print("\n" + "=" * 75)
    print("BRACKET CONTENTS")
    print("=" * 75)

    for region_id in sorted(r64["derived_region"].unique()):
        region_games = r64[r64["derived_region"] == region_id]
        print(f"\n  REGION {region_id + 1}")
        print(f"  {'Loc':>4}  {'S':>2}  {'Home':<28}  vs  {'Away':<28}  {'S':>2}")
        print("  " + "-" * 72)
        for _, row in region_games.iterrows():
            home   = str(row["home_team_name"])[:27] if pd.notna(row["home_team_name"]) else "TBD"
            away   = str(row["away_team_name"])[:27] if pd.notna(row["away_team_name"]) else "TBD"
            h_seed = int(row["home_seed"]) if pd.notna(row["home_seed"]) else "?"
            a_seed = int(row["away_seed"]) if pd.notna(row["away_seed"]) else "?"
            loc    = int(row["bracket_location"]) if pd.notna(row["bracket_location"]) else "?"
            print(f"  {loc:>4}  {h_seed:>2}  {home:<28}  vs  {away:<28}  {a_seed:>2}")

    playin = df[df["round"] == 0]
    if not playin.empty:
        print(f"\n  FIRST FOUR")
        print(f"  {'S':>2}  {'Home':<28}  vs  {'Away':<28}")
        print("  " + "-" * 65)
        for _, row in playin.iterrows():
            home   = str(row["home_team_name"])[:27] if pd.notna(row["home_team_name"]) else "TBD"
            away   = str(row["away_team_name"])[:27] if pd.notna(row["away_team_name"]) else "TBD"
            h_seed = int(row["home_seed"]) if pd.notna(row["home_seed"]) else "?"
            print(f"  {h_seed:>2}  {home:<28}  vs  {away:<28}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull and validate NCAA Tournament bracket"
    )
    parser.add_argument("--season",     type=int, default=2025,
                        help="Season year (2025 = 2025 tournament)")
    parser.add_argument("--model-dir",  default="models",
                        help="Directory to save bracket_raw.parquet")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip printing bracket summary to terminal")
    return parser.parse_args()


def main():
    args      = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    bracket = pull_bracket(args.season)

    is_complete = validate_bracket(bracket)

    if not args.no_summary:
        print_bracket_summary(bracket)

    output_path = model_dir / "bracket_raw.parquet"
    bracket.to_parquet(output_path, index=False)
    log.info("Bracket saved -> %s (%d game slots)", output_path, len(bracket))

    if is_complete:
        log.info("Ready to run: python predict_bracket.py --season %d", args.season)
    else:
        log.warning(
            "Bracket incomplete. Re-run this script later once the API "
            "has populated missing games, then run predict_bracket.py."
        )


if __name__ == "__main__":
    main()