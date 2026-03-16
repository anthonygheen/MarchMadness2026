"""
collect_data.py
---------------
Pulls all training and inference data from the BallDontLie NCAAB API
for a March Madness game-level prediction model.

Data collected:
  - Historical games (2015-present) with scores
  - Team season stats (per-season aggregates)
  - Standings (conference, home/away splits, seeds)
  - AP/Coaches poll rankings
  - March Madness bracket (current year)
  - Betting odds (current tournament)

Output files (data/ directory):
  - games.parquet
  - team_season_stats.parquet
  - standings.parquet
  - rankings.parquet
  - bracket.parquet
  - odds.parquet

Usage:
  export BDL_API_KEY="your_key_here"
  python collect_data.py [--seasons 2015 2016 ... 2024] [--output-dir data]
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://api.balldontlie.io/ncaab/v1"
DEFAULT_SEASONS = list(range(2015, 2025))   # 2015 = 2015-16 season, etc.
DEFAULT_OUTPUT_DIR = "data"
RATE_LIMIT_SLEEP = 0.12   # 600 req/min for GOAT tier → ~10/sec; stay conservative
PER_PAGE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class BDLClient:
    """Thin wrapper around the BallDontLie NCAAB API."""

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(5):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt
                log.warning("Rate limited — sleeping %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Failed after retries: {url}")

    def paginate(self, endpoint: str, params: dict | None = None) -> list[dict]:
        """Cursor-based pagination — returns all records for an endpoint."""
        params = {**(params or {}), "per_page": PER_PAGE}
        records = []
        cursor = None

        while True:
            if cursor is not None:
                params["cursor"] = cursor
            data = self._get(endpoint, params)
            records.extend(data.get("data", []))

            next_cursor = data.get("meta", {}).get("next_cursor")
            if next_cursor is None:
                break
            cursor = next_cursor
            time.sleep(RATE_LIMIT_SLEEP)

        return records


# ---------------------------------------------------------------------------
# Collection functions
# ---------------------------------------------------------------------------

def collect_games(client: BDLClient, seasons: list[int]) -> pd.DataFrame:
    """
    Game-level results — the source of truth for labels (home_win).
    Filters to post-season status so we only keep completed games.
    """
    log.info("Collecting games for seasons: %s", seasons)
    all_records = []

    for season in seasons:
        log.info("  Season %d ...", season)
        records = client.paginate("games", {"seasons[]": season})
        all_records.extend(records)
        time.sleep(RATE_LIMIT_SLEEP)

    df = pd.json_normalize(all_records)

    # Rename nested team cols for clarity
    df = df.rename(columns={
        "home_team.id":           "home_team_id",
        "home_team.college":      "home_team_college",
        "home_team.abbreviation": "home_team_abbr",
        "visitor_team.id":        "away_team_id",
        "visitor_team.college":   "away_team_college",
        "visitor_team.abbreviation": "away_team_abbr",
    })

    # Keep only completed games
    df = df[df["status"] == "post"].copy()

    # Derived label columns
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["point_diff"] = df["home_score"] - df["away_score"]
    df["total_score"] = df["home_score"] + df["away_score"]

    # Clean up datetime
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    keep_cols = [
        "id", "date", "season", "period_detail",
        "home_team_id", "home_team_college", "home_team_abbr",
        "away_team_id",  "away_team_college",  "away_team_abbr",
        "home_score", "away_score", "home_score_h1", "away_score_h1",
        "home_score_h2", "away_score_h2",
        "home_win", "point_diff", "total_score",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    log.info("  → %d completed games", len(df))
    return df


def collect_team_season_stats(client: BDLClient, seasons: list[int]) -> pd.DataFrame:
    """
    Per-team, per-season aggregated stats.
    These become the feature vectors for each team in a matchup.
    """
    log.info("Collecting team season stats ...")
    all_records = []

    for season in seasons:
        log.info("  Season %d ...", season)
        records = client.paginate("team_season_stats", {"season": season})
        for r in records:
            r["season"] = season
        all_records.extend(records)
        time.sleep(RATE_LIMIT_SLEEP)

    df = pd.json_normalize(all_records)

    df = df.rename(columns={
        "team.id":           "team_id",
        "team.college":      "college",
        "team.abbreviation": "abbr",
        "team.conference_id":"conference_id",
    })

    # Derived efficiency / rate stats useful for modeling
    # (raw counts come from API; compute rates where possible)
    if {"fgm", "fga"}.issubset(df.columns):
        df["fg_pct"]  = df["fgm"]  / df["fga"].replace(0, pd.NA)
    if {"fg3m", "fg3a"}.issubset(df.columns):
        df["fg3_pct"] = df["fg3m"] / df["fg3a"].replace(0, pd.NA)
    if {"ftm", "fta"}.issubset(df.columns):
        df["ft_pct"]  = df["ftm"]  / df["fta"].replace(0, pd.NA)

    log.info("  → %d team-season records", len(df))
    return df


def collect_player_season_stats(client: BDLClient, seasons: list[int]) -> pd.DataFrame:
    """
    Player-level season stats — useful for roster-quality features
    (e.g. top-scorer PPG, depth metrics) if you want to go beyond team stats.
    """
    log.info("Collecting player season stats ...")
    all_records = []

    for season in seasons:
        log.info("  Season %d ...", season)
        records = client.paginate("player_season_stats", {"season": season})
        for r in records:
            r["season"] = season
        all_records.extend(records)
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_records:
        return pd.DataFrame()

    df = pd.json_normalize(all_records)

    df = df.rename(columns={
        "team.id":           "team_id",
        "team.college":      "college",
        "player.id":         "player_id",
        "player.first_name": "first_name",
        "player.last_name":  "last_name",
        "player.position":   "position",
    })

    log.info("  → %d player-season records", len(df))
    return df


def collect_standings(client: BDLClient, seasons: list[int]) -> pd.DataFrame:
    """
    Conference standings — win%, home record, away record, playoff_seed.
    Requires iterating over all conference IDs (1-35 covers all active confs).
    """
    log.info("Collecting standings ...")

    # Get all conference IDs dynamically
    conf_data = client._get("conferences")
    conf_ids = [c["id"] for c in conf_data.get("data", [])]

    all_records = []
    for season in seasons:
        log.info("  Season %d ...", season)
        for conf_id in conf_ids:
            try:
                records = client._get(
                    "standings",
                    {"conference_id": conf_id, "season": season}
                ).get("data", [])
                for r in records:
                    r["season"] = season
                all_records.extend(records)
            except requests.HTTPError as e:
                # Some conf/season combos return 404 — skip silently
                if e.response.status_code != 404:
                    log.warning("Standings error conf=%d season=%d: %s", conf_id, season, e)
            time.sleep(RATE_LIMIT_SLEEP)

    df = pd.json_normalize(all_records)

    df = df.rename(columns={
        "team.id":            "team_id",
        "team.college":       "college",
        "team.abbreviation":  "abbr",
        "conference.id":      "conference_id",
        "conference.name":    "conference_name",
    })

    # Parse home/away records into numeric win counts
    for col in ["home_record", "away_record", "conference_record"]:
        if col in df.columns:
            split = df[col].str.split("-", expand=True)
            prefix = col.replace("_record", "")
            df[f"{prefix}_wins"]   = pd.to_numeric(split[0], errors="coerce")
            df[f"{prefix}_losses"] = pd.to_numeric(split[1], errors="coerce")

    log.info("  → %d standing records", len(df))
    return df


def collect_rankings(client: BDLClient, seasons: list[int]) -> pd.DataFrame:
    """
    AP Poll and Coaches Poll weekly rankings.
    We'll later join on the most recent ranking before each game date.
    """
    log.info("Collecting rankings ...")
    all_records = []

    for season in seasons:
        log.info("  Season %d ...", season)
        records = client._get("rankings", {"season": season}).get("data", [])
        for r in records:
            r["season"] = season
        all_records.extend(records)
        time.sleep(RATE_LIMIT_SLEEP)

    df = pd.json_normalize(all_records)

    df = df.rename(columns={
        "team.id":           "team_id",
        "team.college":      "college",
        "team.abbreviation": "abbr",
    })

    log.info("  → %d ranking records", len(df))
    return df


def collect_bracket(client: BDLClient, year: int) -> pd.DataFrame:
    """
    Current tournament bracket — seeds, regions, matchups.
    This is the inference target: which teams are actually in the field.
    """
    log.info("Collecting %d bracket ...", year)
    try:
        data = client._get("bracket", {"season": year - 1})  # API season = start year
        records = data.get("data", [])
    except requests.HTTPError as e:
        log.warning("Bracket not yet available: %s", e)
        return pd.DataFrame()

    df = pd.json_normalize(records)
    log.info("  → %d bracket entries", len(df))
    return df


def collect_odds(client: BDLClient, year: int) -> pd.DataFrame:
    """
    Betting odds — useful for calibration and as an additional feature
    (market-implied probability is a strong signal).
    """
    log.info("Collecting odds for %d tournament ...", year)
    try:
        records = client.paginate("odds", {"season": year - 1})
    except requests.HTTPError as e:
        log.warning("Odds endpoint error: %s", e)
        return pd.DataFrame()

    df = pd.json_normalize(records)
    log.info("  → %d odds records", len(df))
    return df


# ---------------------------------------------------------------------------
# Feature engineering helpers (pre-model)
# ---------------------------------------------------------------------------

def build_team_feature_lookup(
    team_season_stats: pd.DataFrame,
    standings: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges team season stats, standings, and final-week rankings into a
    single lookup table keyed on (team_id, season).

    This is the feature matrix you'll join onto each game row (twice —
    once for home team, once for away team).
    """
    log.info("Building team feature lookup ...")

    # Start from season stats as the base
    base = team_season_stats.copy()

    # --- Standings merge ---
    # Keep only columns that add signal beyond what season stats provide
    standing_cols = [
        "team_id", "season",
        "win_percentage", "conference_win_percentage",
        "home_wins", "home_losses", "away_wins", "away_losses",
        "conference_wins", "conference_losses",
        "playoff_seed", "conference_name",
    ]
    standing_cols = [c for c in standing_cols if c in standings.columns]
    base = base.merge(
        standings[standing_cols],
        on=["team_id", "season"],
        how="left",
        suffixes=("", "_std"),
    )

    # --- Rankings merge — use last (highest-week) ranking per team/season ---
    if not rankings.empty and "week" in rankings.columns:
        latest_rank = (
            rankings.sort_values("week")
            .groupby(["team_id", "season", "poll"])
            .last()
            .reset_index()
        )
        # Pivot so we get ap_rank and coach_rank columns
        ap_rank = (
            latest_rank[latest_rank["poll"] == "ap"]
            [["team_id", "season", "rank"]]
            .rename(columns={"rank": "ap_rank"})
        )
        coach_rank = (
            latest_rank[latest_rank["poll"] == "coach"]
            [["team_id", "season", "rank"]]
            .rename(columns={"rank": "coach_rank"})
        )
        base = base.merge(ap_rank,    on=["team_id", "season"], how="left")
        base = base.merge(coach_rank, on=["team_id", "season"], how="left")

        # Unranked teams get a value just below 25 for ordinal use
        base["ap_rank"]    = base["ap_rank"].fillna(26).astype(float)
        base["coach_rank"] = base["coach_rank"].fillna(26).astype(float)
        base["is_ranked"]  = (base["ap_rank"] <= 25).astype(int)

    log.info("  → feature lookup shape: %s", base.shape)
    return base


def build_game_dataset(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Joins team features onto each game row to produce a model-ready dataset.

    Each row = one game.
    Features are expressed as home_<stat> and away_<stat>, plus derived
    differential columns (diff_<stat> = home - away) for linear models.

    Target: home_win (binary) and point_diff (continuous).
    """
    log.info("Building game-level dataset ...")

    stat_cols = [c for c in team_features.columns if c not in
                 ("team_id", "season", "college", "abbr",
                  "conference_id", "conference_name")]

    home_feat = team_features[["team_id", "season"] + stat_cols].add_prefix("home_")
    home_feat = home_feat.rename(columns={
        "home_team_id": "home_team_id",
        "home_season":  "season",
    })
    home_feat.columns = [
        "home_team_id" if c == "home_team_id" else
        "season"       if c == "home_season"  else c
        for c in home_feat.columns
    ]

    away_feat = team_features[["team_id", "season"] + stat_cols].add_prefix("away_")
    away_feat.columns = [
        "away_team_id" if c == "away_team_id" else
        "season"       if c == "away_season"  else c
        for c in away_feat.columns
    ]

    # Rebuild cleanly to avoid the prefix rename mess
    home_feat = team_features[["team_id", "season"] + stat_cols].copy()
    home_feat = home_feat.rename(columns={"team_id": "home_team_id"})
    home_feat = home_feat.rename(columns={c: f"home_{c}" for c in stat_cols})

    away_feat = team_features[["team_id", "season"] + stat_cols].copy()
    away_feat = away_feat.rename(columns={"team_id": "away_team_id"})
    away_feat = away_feat.rename(columns={c: f"away_{c}" for c in stat_cols})

    df = games.merge(home_feat, on=["home_team_id", "season"], how="left")
    df = df.merge(away_feat,   on=["away_team_id", "season"],  how="left")

    # Add differential columns — these tend to be among the most predictive features
    for col in stat_cols:
        h, a = f"home_{col}", f"away_{col}"
        if h in df.columns and a in df.columns:
            try:
                df[f"diff_{col}"] = df[h] - df[a]
            except TypeError:
                pass  # skip non-numeric cols

    log.info("  → game dataset shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, path: Path, name: str) -> None:
    if df.empty:
        log.warning("  %s is empty — skipping save", name)
        return
    path.mkdir(parents=True, exist_ok=True)
    fp = path / f"{name}.parquet"
    df.to_parquet(fp, index=False)
    log.info("  Saved %s → %s (%d rows, %d cols)", name, fp, len(df), df.shape[1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="BDL NCAAB data collector")
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=DEFAULT_SEASONS,
        help="List of season start years to collect (e.g. 2018 2019 2020)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to write parquet files",
    )
    parser.add_argument(
        "--tournament-year", type=int, default=2025,
        help="Year of the tournament to predict (e.g. 2025 for March 2025)",
    )
    parser.add_argument(
        "--skip-player-stats", action="store_true",
        help="Skip player season stats (large payload, optional for v1 model)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("BDL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set BDL_API_KEY environment variable before running.\n"
            "  export BDL_API_KEY='your_key_here'"
        )

    client = BDLClient(api_key)
    output = Path(args.output_dir)
    seasons = args.seasons
    t_year = args.tournament_year

    log.info("Starting data collection — seasons %d–%d, tournament %d",
             min(seasons), max(seasons), t_year)

    # ------------------------------------------------------------------
    # Raw pulls
    # ------------------------------------------------------------------
    games             = collect_games(client, seasons)
    team_season_stats = collect_team_season_stats(client, seasons)
    standings         = collect_standings(client, seasons)
    rankings          = collect_rankings(client, seasons)
    bracket           = collect_bracket(client, t_year)
    odds              = collect_odds(client, t_year)

    player_season_stats = pd.DataFrame()
    if not args.skip_player_stats:
        player_season_stats = collect_player_season_stats(client, seasons)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------
    team_features = build_team_feature_lookup(team_season_stats, standings, rankings)
    game_dataset  = build_game_dataset(games, team_features)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save(games,               output, "games")
    save(team_season_stats,   output, "team_season_stats")
    save(player_season_stats, output, "player_season_stats")
    save(standings,           output, "standings")
    save(rankings,            output, "rankings")
    save(bracket,             output, "bracket")
    save(odds,                output, "odds")
    save(team_features,       output, "team_features")
    save(game_dataset,        output, "game_dataset")

    log.info("Done. All files written to %s/", args.output_dir)
    log.info("Next step: python train_model.py --data-dir %s", args.output_dir)


if __name__ == "__main__":
    main()