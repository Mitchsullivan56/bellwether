# ---------------------------------------------------------------
# build_dataset.py  —  Bellwether dataset + baselines
# USAGE: python3 build_dataset.py
# Pulls weekly player stats (nflreadpy), builds a leakage-free
# feature table for completed games (model_table.parquet), AND a
# target-less feature row for every active player in the next
# unplayed week (projection_frame.parquet) so the model can project
# an upcoming lineup. Prints baseline MAE per position.
# ---------------------------------------------------------------

import pathlib
import numpy as np
import pandas as pd
import nflreadpy as nfl
import ff_seasons

# --- CONFIG ---
SEASONS   = ff_seasons.resolve_seasons()   # 2020..current (auto-extends each season)
POSITIONS = ["QB", "RB", "WR", "TE"]
TARGET    = "fantasy_points_ppr"        # switch to "fantasy_points" for standard scoring
OUT_PATH  = pathlib.Path(__file__).resolve().parent / "data" / "model_table.parquet"
PROJ_PATH = pathlib.Path(__file__).resolve().parent / "data" / "projection_frame.parquet"

# Volume stats we'll roll into trailing features (only those present are used)
VOLUME_COLS = [
    "targets", "receptions", "receiving_yards", "carries", "rushing_yards",
    "passing_yards", "attempts", "completions", "passing_tds",
    "rushing_tds", "receiving_tds", "target_share", "air_yards_share",
]


def add_upcoming_rows(df, sched_reg, present_volume):
    """Append target-less rows for every active player in the next scheduled
    (unplayed) week so they flow through the same feature pipeline below.
    Returns df unchanged in the offseason / when the season is complete."""
    cur_season = int(df["season"].max())
    cur = df[df["season"] == cur_season]
    if cur.empty:
        return df
    upcoming_week = int(cur["week"].max()) + 1
    games = sched_reg[(sched_reg["season"] == cur_season) &
                      (sched_reg["week"] == upcoming_week)]
    if games.empty:
        return df  # nothing scheduled next — offseason or season complete
    opp = {}
    for _, g in games.iterrows():
        opp[g["home_team"]] = g["away_team"]
        opp[g["away_team"]] = g["home_team"]
    # each player's most recent team / position / name this season
    latest = (cur.sort_values("week").groupby("player_id").tail(1)
              [["player_id", "player_display_name", "position", "team"]].copy())
    latest = latest[latest["team"].isin(opp)]          # drop teams on bye
    latest["season"] = cur_season
    latest["week"] = upcoming_week
    latest["opponent_team"] = latest["team"].map(opp)
    latest[TARGET] = np.nan
    for c in present_volume:
        latest[c] = np.nan
    latest = latest[df.columns]
    print(f"  + {len(latest)} upcoming-week rows ({cur_season} Wk{upcoming_week}) "
          f"synthesized for forward projection")
    out = pd.concat([df, latest], ignore_index=True)
    return out.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


# --- LOAD ---
print(f"Loading weekly player stats for {SEASONS[0]}-{SEASONS[-1]}...")
df = nfl.load_player_stats(seasons=SEASONS).to_pandas()

# Keep regular-season skill-position rows that actually scored a target value
if "season_type" in df.columns:
    df = df[df["season_type"] == "REG"]
df = df[df["position"].isin(POSITIONS)].copy()
df = df.dropna(subset=[TARGET])

# Stable game order per player
df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

present_volume = [c for c in VOLUME_COLS if c in df.columns]
keep = ["player_id", "player_display_name", "position", "season", "week",
        "team", "opponent_team", TARGET] + present_volume
keep = [c for c in keep if c in df.columns]
df = df[keep].copy()

# --- LOAD SCHEDULE (Vegas lines + matchups, for features and forward projection) ---
print("Loading schedules (Vegas lines, weather, rest, matchups)...")
sched_reg = nfl.load_schedules(seasons=SEASONS).to_pandas()
sched_reg = sched_reg[sched_reg["game_type"] == "REG"].copy()

# Synthesize the upcoming (unplayed) week before any features are built.
df = add_upcoming_rows(df, sched_reg, present_volume)

# --- VEGAS LINES + WEATHER + REST (pre-game info, known before kickoff -> no shift) ---
sched = sched_reg.dropna(subset=["spread_line", "total_line"])

# spread_line is signed from the home team's perspective (positive = home favored).
# implied team score = half the total, adjusted by half the spread in their favor.
ctx_cols = ["season", "week", "total_line", "spread_line", "roof", "temp", "wind"]
home = sched[ctx_cols + ["home_team", "home_rest"]].rename(
    columns={"home_team": "team", "home_rest": "rest_days"})
home["team_implied_total"] = (home["total_line"] + home["spread_line"]) / 2
home["opp_implied_total"] = (home["total_line"] - home["spread_line"]) / 2
home["team_spread"] = home["spread_line"]
home["is_home"] = 1

away = sched[ctx_cols + ["away_team", "away_rest"]].rename(
    columns={"away_team": "team", "away_rest": "rest_days"})
away["team_implied_total"] = (away["total_line"] - away["spread_line"]) / 2
away["opp_implied_total"] = (away["total_line"] + away["spread_line"]) / 2
away["team_spread"] = -away["spread_line"]
away["is_home"] = 0

vegas_cols = ["season", "week", "team", "total_line", "team_implied_total",
              "opp_implied_total", "team_spread", "roof", "temp", "wind",
              "rest_days", "is_home"]
vegas = pd.concat([home[vegas_cols], away[vegas_cols]], ignore_index=True).rename(
    columns={"total_line": "game_total"})

# Indoor games have no wind and a controlled temperature -> impute the
# physical reality instead of leaving it as a missing forecast.
indoor = vegas["roof"].isin(["dome", "closed"])
vegas.loc[indoor, "wind"] = vegas.loc[indoor, "wind"].fillna(0)
vegas.loc[indoor, "temp"] = vegas.loc[indoor, "temp"].fillna(68)

df = df.merge(vegas, on=["season", "week", "team"], how="left")

# --- SNAP SHARE (post-game result, like the volume stats -> trailing only) ---
print("Loading snap counts...")
ids = nfl.load_ff_playerids().to_pandas()[["pfr_id", "gsis_id"]].dropna()
snaps = nfl.load_snap_counts(seasons=SEASONS).to_pandas()
snaps = snaps[snaps["game_type"] == "REG"]
snaps = snaps.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
snaps = snaps[["gsis_id", "season", "week", "offense_pct"]].rename(columns={"gsis_id": "player_id"})
snaps = snaps.drop_duplicates(subset=["player_id", "season", "week"])
df = df.merge(snaps, on=["player_id", "season", "week"], how="left")
present_volume = present_volume + ["offense_pct"]

# --- INJURY STATUS (pre-game report, known before kickoff -> no shift needed) ---
print("Loading injury reports...")
inj = nfl.load_injuries(seasons=SEASONS).to_pandas()
inj = inj[inj["game_type"] == "REG"]
inj = inj[["gsis_id", "season", "week", "report_status"]].rename(
    columns={"gsis_id": "player_id", "report_status": "injury_status"})
inj = inj.drop_duplicates(subset=["player_id", "season", "week"])
df = df.merge(inj, on=["player_id", "season", "week"], how="left")
df["injury_status"] = df["injury_status"].fillna("Healthy")

# --- NEXT GEN STATS (post-game result, like the volume stats -> trailing only) ---
print("Loading Next Gen Stats...")
NGS_COLS = {
    "passing": ["avg_time_to_throw", "completion_percentage_above_expectation",
                "aggressiveness", "avg_air_yards_differential"],
    "rushing": ["efficiency", "rush_yards_over_expected_per_att",
                "percent_attempts_gte_eight_defenders"],
    "receiving": ["avg_separation", "avg_cushion", "avg_yac_above_expectation",
                  "percent_share_of_intended_air_yards"],
}
for stat_type, cols in NGS_COLS.items():
    ngs = nfl.load_nextgen_stats(stat_type=stat_type, seasons=SEASONS).to_pandas()
    ngs = ngs[(ngs["season_type"] == "REG") & (ngs["week"] > 0)]  # week 0 = season aggregate row
    ngs = ngs[["player_gsis_id", "season", "week"] + cols].rename(
        columns={"player_gsis_id": "player_id"})
    ngs = ngs.drop_duplicates(subset=["player_id", "season", "week"])
    df = df.merge(ngs, on=["player_id", "season", "week"], how="left")
    present_volume = present_volume + cols

# --- EXPECTED FANTASY POINTS FROM OPPORTUNITY (post-game result -> trailing only) ---
# nflverse ff_opportunity models the fantasy points a player "earned" from
# volume / air yards / down-distance, independent of whether the TDs actually
# fell. Trailing expected points is a sharper read on a player's role than raw
# past scoring, and the gap (actual - expected) flags TD luck that's likely to
# regress. This is exactly the opportunity signal name-brand consensus
# rankings underweight, so it's our clearest differentiator.
print("Loading expected fantasy points (ff_opportunity)...")
opp = nfl.load_ff_opportunity(seasons=SEASONS).to_pandas()
opp = opp[["player_id", "season", "week", "total_fantasy_points_exp",
           "total_fantasy_points_diff"]].rename(columns={
    "total_fantasy_points_exp": "fp_exp",
    "total_fantasy_points_diff": "fp_over_exp"})
opp["season"] = opp["season"].astype(int)
opp["week"] = opp["week"].astype(int)
opp = opp.drop_duplicates(subset=["player_id", "season", "week"])
df = df.merge(opp, on=["player_id", "season", "week"], how="left")
present_volume = present_volume + ["fp_exp", "fp_over_exp"]

# --- OPPONENT DEFENSE ALLOWED (leakage-free: trailing avg of points the
# opponent has allowed to this position, not bounded by season so week 1
# inherits a prior from the end of last season instead of a cold start) ---
print("Computing opponent defense strength allowed by position...")
def_allowed = (df.groupby(["season", "week", "opponent_team", "position"])[TARGET]
               .mean().reset_index()
               .rename(columns={TARGET: "pts_allowed", "opponent_team": "def_team"})
               .sort_values(["def_team", "position", "season", "week"]))
def_grp = def_allowed.groupby(["def_team", "position"], sort=False)
def_allowed["def_pts_allowed_avg3"] = def_grp["pts_allowed"].transform(
    lambda s: s.rolling(3, min_periods=1).mean().shift(1))
def_allowed = def_allowed.rename(columns={"def_team": "opponent_team"})
df = df.merge(
    def_allowed[["season", "week", "opponent_team", "position", "def_pts_allowed_avg3"]],
    on=["season", "week", "opponent_team", "position"], how="left")

# --- FEATURES (leakage-free: weeks 1..w-1 only) ---
# Each feature is built per (player, season) with groupby-transform so the
# grouping keys stay as columns. rolling/expanding then .shift(1) guarantees
# week w never sees its own stats.
grp = df.groupby(["player_id", "season"], sort=False)

df["pts_last1"]    = grp[TARGET].shift(1)
df["pts_avg3"]     = grp[TARGET].transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
df["pts_avg5"]     = grp[TARGET].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
df["pts_season"]   = grp[TARGET].transform(lambda s: s.expanding().mean().shift(1))
df["games_so_far"] = grp.cumcount()        # games played before this week

for col in present_volume:
    df[f"{col}_avg3"] = grp[col].transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))

# Drop the raw same-week volume stats (and team, used only for the Vegas join)
# — model only sees trailing versions plus the pre-game Vegas/injury features.
df = df.drop(columns=present_volume + ["team"], errors="ignore")

# --- SPLIT: completed rows (training/eval) vs the upcoming week (projection) ---
upcoming = df[df[TARGET].isna()].copy()
df = df[df[TARGET].notna()].reset_index(drop=True)

# --- BASELINES ---
# Two honest reference points the model must beat:
#   naive_last : predict last game's points
#   avg3       : predict trailing 3-game average
# Scored only where a 3-game history exists, so it's a fair bar.
eval_df = df[df["games_so_far"] >= 3].copy()
eval_df["err_naive"] = (eval_df[TARGET] - eval_df["pts_last1"]).abs()
eval_df["err_avg3"]  = (eval_df[TARGET] - eval_df["pts_avg3"]).abs()

print("\nBASELINE MAE  (lower = harder to beat)")
print("-" * 46)
print(f"  {'Position':<10}{'naive_last':>12}{'avg3':>12}{'n':>10}")
print("-" * 46)
for pos in POSITIONS:
    sub = eval_df[eval_df["position"] == pos]
    if sub.empty:
        continue
    print(f"  {pos:<10}{sub['err_naive'].mean():>12.2f}"
          f"{sub['err_avg3'].mean():>12.2f}{len(sub):>10}")
print("-" * 46)
print(f"  {'ALL':<10}{eval_df['err_naive'].mean():>12.2f}"
      f"{eval_df['err_avg3'].mean():>12.2f}{len(eval_df):>10}")
print("-" * 46)

# --- SAVE ---
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f"\nModel table saved: {OUT_PATH}  ({len(df):,} rows, {df.shape[1]} cols)")

# --- SAVE PROJECTION FRAME (the upcoming week, for the live lineup board) ---
if len(upcoming):
    upcoming.to_parquet(PROJ_PATH, index=False)
    uw = f"{int(upcoming['season'].iloc[0])} Wk{int(upcoming['week'].iloc[0])}"
    print(f"Projection frame saved: {PROJ_PATH}  ({len(upcoming)} players, {uw})")
elif PROJ_PATH.exists():
    PROJ_PATH.unlink()  # stale (offseason) — drop so the app falls back to demo
    print("No upcoming week to project (offseason / season complete); cleared old frame.")
else:
    print("No upcoming week to project (offseason / season complete).")

# --- SANITY CHECK: top scorers, most recent completed week ---
last_season = df["season"].max()
last_week   = df[df["season"] == last_season]["week"].max()
sample = (df[(df["season"] == last_season) & (df["week"] == last_week)]
          .sort_values(TARGET, ascending=False)
          .head(10)[["player_display_name", "position", TARGET,
                     "pts_avg3", "pts_season"]])
print(f"\nTop 10 - {last_season} Week {int(last_week)} "
      f"(actual vs what the baselines would have guessed):")
print(sample.to_string(index=False))
