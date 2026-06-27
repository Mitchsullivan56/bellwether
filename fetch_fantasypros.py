# ---------------------------------------------------------------
# fetch_fantasypros.py  —  FF Projections, expert rankings archive
# USAGE: python3 fetch_fantasypros.py
# Pulls FantasyPros weekly expert-consensus rankings (ECR) from the
# DynastyProcess/nflverse archive, maps each ranking snapshot to the
# NFL season/week it was published for, and crosswalks FantasyPros
# player IDs to gsis_id (the player_id used in data/model_table.parquet).
# ---------------------------------------------------------------

import pathlib
import pandas as pd
import nflreadpy as nfl
import ff_seasons

OUT_PATH = pathlib.Path(__file__).resolve().parent / "data" / "fp_rankings.parquet"
SEASONS = ff_seasons.resolve_seasons()   # 2020..current (auto-extends each season)
WEEKLY_PAGES = {"weekly-qb": "QB", "weekly-rb": "RB", "weekly-wr": "WR", "weekly-te": "TE"}

# --- LOAD ---
print("Loading FantasyPros weekly rankings archive...")
rankings = nfl.load_ff_rankings(type="all").to_pandas()
rankings = rankings[rankings["page_type"].isin(WEEKLY_PAGES)].copy()
rankings["position"] = rankings["page_type"].map(WEEKLY_PAGES)
rankings["scrape_date"] = pd.to_datetime(rankings["scrape_date"])

print("Loading player ID crosswalk...")
ids = nfl.load_ff_playerids().to_pandas()
ids = ids[["fantasypros_id", "gsis_id"]].dropna()
ids["fantasypros_id"] = ids["fantasypros_id"].astype(str)
rankings["id"] = rankings["id"].astype(str)

rankings = rankings.merge(ids, left_on="id", right_on="fantasypros_id", how="inner")

print("Loading game schedule to map scrape_date -> season/week...")
sched = nfl.load_schedules(seasons=SEASONS).to_pandas()
sched = sched[sched["game_type"] == "REG"]
week_starts = (sched.groupby(["season", "week"])["gameday"]
               .min().reset_index())
week_starts["gameday"] = pd.to_datetime(week_starts["gameday"])
week_starts = week_starts.sort_values("gameday").reset_index(drop=True)

# Each ranking snapshot is published a few days before that week's games,
# so assign it to the next upcoming week's game date.
rankings = rankings.sort_values("scrape_date")
rankings = pd.merge_asof(
    rankings, week_starts,
    left_on="scrape_date", right_on="gameday",
    direction="forward",
)
rankings = rankings.dropna(subset=["season", "week"])

keep = rankings[["gsis_id", "season", "week", "position", "ecr"]].rename(
    columns={"gsis_id": "player_id"}
)
keep["season"] = keep["season"].astype(int)
keep["week"] = keep["week"].astype(int)

# A player can appear on more than one ranking page per week (e.g. flex);
# keep the most specific positional ranking, lowest scrape-to-week gap wins ties.
keep = keep.drop_duplicates(subset=["player_id", "season", "week", "position"])

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
keep.to_parquet(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  ({len(keep):,} rows)")
print(keep.groupby("season")["week"].nunique())
print(f"\nPositions covered: {keep['position'].unique().tolist()}")
